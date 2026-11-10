"""Compare released Touch2Touch checkpoints on a directory of paired .pt files."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.functional import interpolate


REPO = Path(__file__).resolve().parents[1]
DIFFUSION = REPO / "touch2touch/stable_diffusion"
FINGERS = ("right", "left")
VQ_WEIGHTS = (
    REPO / "scripts/models"
    / "VQ-VAE-small_dataset_new_partial_data_cross_GB_mod_4_single_random_sensor_color_jitter_rotation_flipping_run_model"
    / "best.pt"
)
SEED = 42
PEAK_DEPTH = 0.0225


def time_seconds(value):
    days, separator, clock = value.rpartition("-")
    fields = [int(field) for field in (clock if separator else value).split(":")]
    if len(fields) not in (2, 3):
        raise ValueError(f"Expected Slurm time-left, got {value!r}")
    return (int(days) * 86400 if separator else 0) + sum(
        field * 60 ** i for i, field in enumerate(reversed(fields))
    )


def paired_samples(data):
    gelslim = {p.name for p in (data / "gelslims").glob("*.pt") if p.is_file()}
    bubble = {p.name for p in (data / "bubbles").glob("*.pt") if p.is_file()}
    if not gelslim or gelslim != bubble:
        raise ValueError(
            f"Expected matching GelSlim/Bubble names, got {len(gelslim)} and {len(bubble)}; "
            f"examples with no pair: {sorted(gelslim ^ bubble)[:5]}"
        )
    groups = defaultdict(list)
    for name in sorted(gelslim):
        match = re.fullmatch(r"(.+)_data_\d+\.pt", name)
        if not match:
            raise ValueError(f"Unsupported sample name: {name}")
        groups[match.group(1)].append(name)
    # Take one sample per tool per round so short runs cover every tool.
    return [(tool, name) for batch in zip_longest(*groups.values()) for tool, name
            in zip(groups, batch) if name is not None]


def load(path):
    return torch.load(path, map_location="cpu")


def save_csv(path, rows):
    partial = path.with_suffix(".csv.part")
    try:
        with partial.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def save_json(path, data):
    partial = path.with_suffix(".json.part")
    try:
        partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def run_diffusion(samples, data, output, args, inference_deadline):
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf
    from touch2touch.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
    from touch2touch.stable_diffusion.ldm.util import instantiate_from_config
    from touch2touch.stable_diffusion.scripts.tactile_style_transfer_estimator import TouchEstimator

    class CheckedEstimator(TouchEstimator):
        def load_model(self):
            config = OmegaConf.load(self.opt.config)
            checkpoint = load(self.opt.ckpt)
            model = instantiate_from_config(config.model)
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
            parameter_keys = set(dict(model.named_parameters()))
            required_missing = [
                key for key in incompatible.missing_keys
                if key in parameter_keys or key.startswith("model_ema.") or key == "scale_factor"
                or key.endswith((".running_mean", ".running_var"))
            ]
            if required_missing:
                raise RuntimeError(f"Diffusion checkpoint is missing parameters or EMA state: {required_missing}")
            print(f"Diffusion checkpoint step: {checkpoint.get('global_step')}", flush=True)
            print("Missing non-parameter keys:", incompatible.missing_keys, flush=True)
            print("Unexpected keys:", incompatible.unexpected_keys, flush=True)
            del checkpoint
            self.model = model.to(self.device).eval()
            self.sampler = DDIMSampler(self.model)

    seed_everything(SEED)
    estimator = CheckedEstimator(SimpleNamespace(
        config=str(DIFFUSION / "configs/tactile_style_transfer/gelslim2bubble.yaml"),
        ckpt=str(args.diffusion_weights),
        precision="autocast", plms=False, fixed_code=False, n_samples=1,
        ddim_steps=args.ddim_steps, ddim_eta=0.0, scale=1.0, skip_save=False,
    ))
    test = output / "test"
    test.mkdir()
    saved = test / "diffusion_results" / args.diffusion_weights.parent.name
    completed = {}
    inference_error = None
    next_batch_seconds = 0.0
    for start in range(0, len(samples), args.batch_pairs):
        if time.monotonic() + next_batch_seconds >= inference_deadline:
            print("Stopping diffusion to leave time for evaluation.", flush=True)
            break
        batch = samples[start:start + args.batch_pairs]
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=test, prefix="batch-") as directory:
            directory = Path(directory)
            inputs = directory / "gelslims"
            inputs.mkdir()
            for _, name in batch:
                (inputs / name).symlink_to(data / "gelslims" / name)
            estimator.opt.outdir = str(directory / "previews")
            try:
                for finger in range(len(FINGERS)):
                    estimator.estimate(str(inputs), finger)
                batch_paths = {
                    name: [saved / f"results_samples_{side}" / directory.name / name
                           for side in FINGERS]
                    for _, name in batch
                }
                for paths in batch_paths.values():
                    load_diffusion_predictions(paths)
            except Exception as exc:
                for side in FINGERS:
                    partial = saved / f"results_samples_{side}" / directory.name
                    if partial.is_dir():
                        shutil.rmtree(partial)
                if not completed:
                    raise
                traceback.print_exc()
                inference_error = f"{type(exc).__name__}: {exc}"
                break
            completed.update(batch_paths)
            save_json(output / "completed.json", {
                name: [str(path.relative_to(output)) for path in paths]
                for name, paths in completed.items()
            })
        elapsed = time.monotonic() - started
        next_batch_seconds = max(next_batch_seconds, elapsed * 1.25)
        print(f"Diffusion: {len(completed)}/{len(samples)} pairs, batch {elapsed:.1f}s", flush=True)
    return completed, inference_error


def existing_predictions(samples, source, split):
    if (source / "completed.json").is_file():
        entries = json.loads((source / "completed.json").read_text(encoding="utf-8"))
        completed = {name: [source / path for path in entries[name]]
                     for _, name in samples if name in entries}
    else:
        completed = {}
        for _, name in samples:
            paths = [source / f"results_samples_{side}" / split / name for side in FINGERS]
            if all(path.is_file() for path in paths):
                completed[name] = paths
    if not completed or any(len(paths) != 2 or not all(path.is_file() for path in paths)
                            for paths in completed.values()):
        raise ValueError(f"No valid paired predictions found in {source} for {split}")
    return completed


def preflight(samples, args):
    from touch2touch.vq_vae.modules import model_definition

    vq = model_definition("VQ-VAE-small", 3, 1, 256, 16385, "cpu", single=True).eval()
    vq.load_state_dict(load(args.vq_weights), strict=True)
    stats = load(args.diffusion_weights.with_name("training_stats.pt"))
    for key in ("gelslim_mean", "gelslim_std", "bubbles_min", "bubbles_max"):
        if key not in stats or not torch.isfinite(torch.as_tensor(stats[key])).all():
            raise ValueError(f"Invalid training statistic: {key}")
    if (torch.as_tensor(stats["gelslim_mean"]).numel() != 3
            or torch.as_tensor(stats["gelslim_std"]).numel() != 3
            or not (torch.as_tensor(stats["gelslim_std"]) > 0).all()
            or torch.as_tensor(stats["bubbles_min"]).numel() != 1
            or torch.as_tensor(stats["bubbles_max"]).numel() != 1
            or float(stats["bubbles_max"]) <= float(stats["bubbles_min"])):
        raise ValueError("Invalid normalization dimensions, standard deviation or depth bounds")
    # Inspect every input before spending GPU time; no model forward is needed.
    for _, name in samples:
        bubble = load(args.data / "bubbles" / name)["bubble_imprint"]
        gel = load(args.data / "gelslims" / name)
        if bubble.shape != (2, 1, 140, 175) or not torch.isfinite(bubble).all():
            raise ValueError(f"Invalid Bubble input: {name}")
        image, reference = gel["gelslim"], gel["gelslim_ref"]
        if (image.ndim != 4 or image.shape[:2] != (2, 3)
                or image.shape != reference.shape or min(image.shape[-2:]) < 1
                or not torch.is_floating_point(image) or not torch.is_floating_point(reference)
                or not torch.isfinite(image).all() or not torch.isfinite(reference).all()):
            raise ValueError(f"Invalid GelSlim input or reference (expected floating tensors): {name}")
    _, sample = samples[0]
    gt = load(args.data / "bubbles" / sample)["bubble_imprint"].float()
    gelslim = load(args.data / "gelslims" / sample)
    if gt.shape != (2, 1, 140, 175) or not torch.isfinite(gt).all():
        raise ValueError(f"Invalid Bubble sample: {sample}")
    with torch.inference_mode():
        pred = vqvae_depth(vq, gelslim, gt.shape[-2:], "cpu")
        if pred.shape != gt.shape or not torch.isfinite(pred).all():
            raise ValueError(f"Invalid VQ-VAE prediction: {sample}")
        print("VQ-VAE CPU sample metrics:", pixel_metrics(gt, pred, "cpu"))
    print("Preflight: matched sample names:", len(samples), flush=True)
    if args.preflight_only:
        from omegaconf import OmegaConf
        from touch2touch.stable_diffusion.scripts.tactile_style_transfer_estimator import TouchEstimator

        config = OmegaConf.load(DIFFUSION / "configs/tactile_style_transfer/gelslim2bubble.yaml")
        from touch2touch.stable_diffusion.ldm.util import get_obj_from_str
        for section in (config.model, config.model.params.unet_config,
                        config.model.params.first_stage_config, config.model.params.cond_stage_config):
            get_obj_from_str(section.target)

        if not args.skip_fid:
            from pytorch_fid.inception import InceptionV3

            model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
            with tempfile.TemporaryDirectory(prefix="touch2touch-fid-check-") as directory:
                features = fid_features(gt, model, "cpu", Path(directory))
            if features.shape != (2, 2048) or not np.isfinite(features).all():
                raise RuntimeError("FID CPU preflight produced invalid features")
        print("PASS: imports, VQ-VAE, CPU metric path and input inventory", flush=True)


def resize(image, size):
    return interpolate(image, size=size, mode="bilinear", align_corners=False)


def load_diffusion_predictions(paths):
    predictions = [load(path)["prediction"].float() for path in paths]
    if len(predictions) != 2 or any(pred.shape != (1, 3, 256, 256)
                                     or not torch.isfinite(pred).all() for pred in predictions):
        raise ValueError(f"Invalid pair of diffusion predictions: {paths}")
    return predictions


def diffusion_depth(paths, size, bubbles_max):
    predictions = load_diffusion_predictions(paths)
    depth = torch.cat(predictions).mean(1, keepdim=True)
    return resize(2 * depth - bubbles_max, size).flip((-2, -1))


def vqvae_depth(model, gelslim, size, device):
    mean = torch.tensor([-0.0082, -0.0059, -0.0066]).view(1, 3, 1, 1)
    std = torch.tensor([0.0989, 0.0746, 0.0731]).view(1, 3, 1, 1)
    image = (gelslim["gelslim"] - gelslim["gelslim_ref"]) / 255
    x = (resize(image, (128, 128)) - mean) / std
    pred, _, _ = model(x.to(device))
    return resize(pred.cpu(), size).flip((-2, -1)) * 0.00424 + 0.00382


def pixel_metrics(gt, pred, device):
    from touch2touch.evaluation.ssim import SSIM

    error = pred - gt
    mse = error.square().mean().item()
    psnr = 100.0 if mse == 0 else 20 * math.log10(PEAK_DEPTH / math.sqrt(mse))
    data_range = (pred.max() - pred.min()).item()
    if data_range <= 0:
        raise ValueError("Cannot score SSIM for a prediction with zero depth range")
    ssim = SSIM(L=data_range).to(device)
    total = sum(ssim(gt_chunk.to(device), pred_chunk.to(device)).item() * len(gt_chunk)
                for gt_chunk, pred_chunk in zip(gt.split(16), pred.split(16)))
    return {"psnr_db": psnr, "ssim": total / len(gt), "mae": error.abs().mean().item(), "mse": mse}


def frechet(a, b):
    # The trace of the covariance square root equals the sum of singular
    # values of the cross-product of the centered, scaled feature matrices.
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]
            or min(len(a), len(b)) < 2 or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError("FID requires finite feature arrays with matching dimensions and at least two images")
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean_a, mean_b = a.mean(0), b.mean(0)
    a = (a - mean_a) / math.sqrt(len(a) - 1)
    b = (b - mean_b) / math.sqrt(len(b) - 1)
    singular = np.linalg.svd(a @ b.T, compute_uv=False)
    return max(0.0, float(np.square(mean_a - mean_b).sum()
                          + np.square(a).sum() + np.square(b).sum() - 2 * singular.sum()))


def fid_features(images, model, device, parent):
    import cv2
    from pytorch_fid.fid_score import get_activations

    with tempfile.TemporaryDirectory(dir=parent, prefix="fid-") as directory:
        files = []
        for index, image in enumerate(images):
            path = Path(directory) / f"{index:06d}.jpg"
            pixels = cv2.normalize(image.squeeze(0).numpy(), None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if not cv2.imwrite(str(path), pixels):
                raise OSError(f"Cannot write FID input: {path}")
            files.append(path)
        return get_activations(files, model, batch_size=32, dims=2048,
                               device=device, num_workers=0)


@torch.inference_mode()
def evaluate(samples, completed, data, output, args, device):
    from touch2touch.vq_vae.modules import model_definition

    vq = model_definition("VQ-VAE-small", 3, 1, 256, 16385, device, single=True).eval()
    vq.load_state_dict(load(args.vq_weights), strict=True)
    bubbles_max = float(load(args.diffusion_weights.with_name("training_stats.pt"))["bubbles_max"])
    vq_directory = output / "vqvae"
    vq_directory.mkdir()
    rows, groups, truth = [], defaultdict(list), []
    predictions = {"diffusion": [], "vq_vae": []}
    for tool, name in samples:
        if name not in completed:
            continue
        gt = load(data / "bubbles" / name)["bubble_imprint"].float()
        if gt.shape != (2, 1, 140, 175) or not torch.isfinite(gt).all():
            raise ValueError(f"Invalid ground truth: {name}")
        outputs = {
            "diffusion": diffusion_depth(completed[name], gt.shape[-2:], bubbles_max),
            "vq_vae": vqvae_depth(vq, load(data / "gelslims" / name), gt.shape[-2:], device),
        }
        groups[tool].extend((2 * len(truth), 2 * len(truth) + 1))
        truth.append(gt)
        for method, pred in outputs.items():
            if pred.shape != gt.shape or not torch.isfinite(pred).all():
                raise ValueError(f"Invalid {method} output: {name}")
            predictions[method].append(pred)
            for finger, side in enumerate(FINGERS):
                error = pred[finger] - gt[finger]
                rows.append({"sample": name, "tool": tool, "finger": side, "model": method,
                             "mae": error.abs().mean().item(), "mse": error.square().mean().item()})
        torch.save({"prediction": outputs["vq_vae"], "fingers": FINGERS}, vq_directory / name)
    del vq
    torch.cuda.empty_cache()
    save_csv(output / "per_image.csv", rows)
    gt = torch.cat(truth)
    predictions = {name: torch.cat(tensors) for name, tensors in predictions.items()}
    grouped = {"all_completed": list(range(len(gt))), **groups}
    summary = []
    for group, indices in grouped.items():
        for method, pred in predictions.items():
            summary.append({"group": group, "model": method, "pairs": len(indices) // 2,
                            **pixel_metrics(gt[indices], pred[indices], device), "fid": ""})
    save_csv(output / "metrics.csv", summary)
    fid_error = None
    if not args.skip_fid:
        try:
            from pytorch_fid.inception import InceptionV3

            inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
            features = {method: fid_features(batch, inception, device, output)
                        for method, batch in {"gt": gt, **predictions}.items()}
            for row in summary:
                ids = grouped[row["group"]]
                row["fid"] = frechet(features["gt"][ids], features[row["model"]][ids])
        except Exception as exc:
            print(f"FID failed; pixel metrics are available: {exc}", file=sys.stderr)
            fid_error = str(exc)
            for row in summary:
                row["fid"] = ""
    for method in predictions:
        tools = [row for row in summary if row["model"] == method and row["group"] in groups]
        summary.append({"group": "mean_over_tools", "model": method, "pairs": len(completed),
                        **{metric: float(np.mean([row[metric] for row in tools]))
                           for metric in ("psnr_db", "ssim", "mae", "mse")},
                        "fid": "" if args.skip_fid or fid_error else
                        float(np.mean([row["fid"] for row in tools]))})
    save_csv(output / "metrics.csv", summary)
    return summary, fid_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/test/test_only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diffusion-weights", type=Path,
                        default=REPO / "checkpoints/diffusion/rot_flip/checkpoint.ckpt")
    parser.add_argument("--vq-weights", type=Path, default=VQ_WEIGHTS)
    parser.add_argument("--ddim-steps", type=int, choices=(10, 200), default=200)
    parser.add_argument("--batch-pairs", type=int, default=12)
    parser.add_argument("--time-left", default="02:00:00")
    parser.add_argument("--reserve-minutes", type=float, default=20)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--expected-pairs", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true",
                       help="Check inputs and metric dependencies on CPU without producing job outputs")
    modes.add_argument("--predictions", type=Path,
                       help="Score existing inference results on CPU (prior job or estimator result root)")
    args = parser.parse_args()
    started = time.monotonic()
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless using --preflight-only")
    # Keep the original checkpoint basename and parent for the estimator.
    args.diffusion_weights = args.diffusion_weights.absolute()
    args.vq_weights = args.vq_weights.resolve()
    if args.diffusion_weights.name != "checkpoint.ckpt":
        parser.error("The estimator requires the diffusion basename checkpoint.ckpt")
    required = [args.vq_weights, args.diffusion_weights.with_name("training_stats.pt")]
    if not args.predictions:
        required.extend((args.diffusion_weights,
                         DIFFUSION / "models/first_stage_models/kl-f8/model.ckpt"))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.data = args.data.resolve()
    samples = paired_samples(args.data)
    if args.expected_pairs is not None and len(samples) != args.expected_pairs:
        raise ValueError(f"Expected {args.expected_pairs} paired inputs, found {len(samples)}")
    if args.preflight_only or args.predictions:
        available = None
    else:
        available = time_seconds(args.time_left)
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is visible in the job")
        if args.batch_pairs < 1 or not 0 <= args.reserve_minutes * 60 < available:
            parser.error("Require a positive batch size and a reserve smaller than the job time left")
    output = args.output.resolve() if args.output is not None else None
    if args.predictions:
        args.predictions = args.predictions.resolve()
    sys.path[:0] = [str(REPO), str(DIFFUSION)]
    preflight(samples, args)
    if args.preflight_only:
        return
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(DIFFUSION)
    metadata = {
        "commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": None if args.predictions else torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "ddim_steps": None if args.predictions else args.ddim_steps,
        "seed": SEED,
        "requested_pairs": len(samples),
        "requested_pairs_by_tool": dict(Counter(tool for tool, _ in samples)),
        "completed_pairs": 0,
        "status": "running",
        "input": str(args.data),
        "prediction_source": str(args.predictions) if args.predictions else None,
        "diffusion_weights": str(args.diffusion_weights),
        "vq_vae_weights": str(args.vq_weights),
        "diffusion_postprocess": "mean over channels, undo repeated depth mapping, resize, rotate 180",
        "train_set_shift": False,
        "official_split": "unknown",
        "paper_comparison": "exploratory only: official split and checkpoint provenance unverified; no train-set Shift or ICP",
        "icp": "not run",
        "fid_preprocess": "per-image min-max, OpenCV JPEG, pytorch-fid InceptionV3 2048",
        "psnr_peak_depth": PEAK_DEPTH,
        "ssim_range": "prediction max minus min per group, as in evaluation/metrics.py",
    }
    manifest = output / "run.json"
    save_json(manifest, metadata)
    inference_error = None
    if args.predictions:
        completed = existing_predictions(samples, args.predictions, args.data.name)
    else:
        completed, inference_error = run_diffusion(samples, args.data, output, args,
                                                   started + available - args.reserve_minutes * 60)
    if not completed:
        raise RuntimeError("No completed pairs before the evaluation reserve")
    torch.cuda.empty_cache()
    metadata["completed_pairs"] = len(completed)
    metadata["inference_seconds"] = None if args.predictions else time.monotonic() - started
    metadata["inference_error"] = inference_error
    metadata["completed_pairs_by_tool"] = dict(Counter(tool for tool, name in samples if name in completed))
    metadata["status"] = "evaluating"
    save_json(manifest, metadata)
    rows, fid_error = evaluate(samples, completed, args.data, output, args,
                               "cpu" if args.predictions else "cuda")
    metadata["status"] = "complete" if len(completed) == len(samples) else "partial"
    metadata["fid_status"] = "failed" if fid_error else "skipped" if args.skip_fid else "complete"
    metadata["fid_error"] = fid_error
    metadata["total_seconds"] = time.monotonic() - started
    save_json(manifest, metadata)
    for row in rows:
        if row["group"] in ("all_completed", "mean_over_tools"):
            print(row)
    print(f"Results: {output} ({metadata['status']})")


if __name__ == "__main__":
    main()
