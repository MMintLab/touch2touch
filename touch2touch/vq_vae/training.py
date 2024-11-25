import os
import glob as glob

import torch
# import pdb; pdb.set_trace()
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset

from touch2touch.vq_vae.datasets_loading import all_datasets_loading, data_symmetry_viz
from touch2touch.vq_vae.modules import model_definition
import wandb

GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])
BUBBLES_MEAN = torch.tensor([0.00382])
BUBBLES_STD = torch.tensor([0.00424])


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Using the GPU!")
else:
  print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

# Datasets
def logging_image_grid(images, captions, step = 0, ncol=7, normalize = True):
    if not normalize:
        norm_text = "_not_normalized"
    else:
        norm_text = ""

    grids = [make_grid(img, nrow=ncol, padding=4, normalize=normalize, scale_each=True) for img in images]
    for grid, caption in zip(grids, captions):
        image = wandb.Image(grid, caption=caption+norm_text)
        wandb.log(data = {caption+norm_text: image}, step = step)
    return

def data_selection():
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    datasets_path = os.path.join(project_path, "data/train")
    bubbles_new_dataset = os.path.join(datasets_path, "bubbles/data")
    gelslim_new_dataset = os.path.join(datasets_path, "gelslims/data")

    bubbles_data_folders = [bubbles_new_dataset]
    gelslim_data_folders = [gelslim_new_dataset]

    test_tools = ['pattern_05_3_lines_angle_2','pattern_35', 'pattern_36']

    return bubbles_data_folders, gelslim_data_folders, test_tools
    

def main(args):
    device = args.device
    print(device)
    # Logging
    if args.debug:
        logging_freq = 1
        project = 'T2T_Debugging'
    else:
        logging_freq = 10
        project = 'T2T'

    wandb.init(
            project=project,
            name=args.output_folder,
            id=args.output_folder,
            resume=False,
        )
    save_filename = './models/{0}'.format(args.output_folder)

    # Dataloading
    gelslim_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD)
                                            ])
    
    bubbles_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(BUBBLES_MEAN, BUBBLES_STD)
                                            ])
    
    bubbles_data_folders, gelslim_data_folders, test_tools = data_selection()

    if args.dataset == 'new':
        all = False
    else:
        all = True

    train_dataset, center_val_dataset, tool_val_datasets, tool_val_names, train_imgs, center_val_imgs, tool_val_imgs = all_datasets_loading(bubbles_data_folders, 
                                                                                                                                            gelslim_data_folders, 
                                                                                                                                            test_tools, 
                                                                                                                                            bubbles_transform, 
                                                                                                                                            gelslim_transform, 
                                                                                                                                            device, 
                                                                                                                                            all=all, 
                                                                                                                                            data=args.data, 
                                                                                                                                            mod=args.mod, 
                                                                                                                                            single=args.single, 
                                                                                                                                            grayscale=args.grayscale, 
                                                                                                                                            cropped=args.cropped, 
                                                                                                                                            distortion=args.gel_distortion, 
                                                                                                                                            random_sensor=args.random_sensor, 
                                                                                                                                            color_jitter=args.color_jitter, 
                                                                                                                                            rotation=args.rotation, 
                                                                                                                                            flipping=args.flipping)
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(center_val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Creation
    if args.data == 'cross_GB':
        num_channels_in = 3
        num_channels_out = 1
    elif args.data == 'bubbles':
        num_channels_in = 1
        num_channels_out = 1
    elif args.data == 'gelslim':
        num_channels_in = 3
        num_channels_out = 3
    elif args.data == 'cross_BG':
        num_channels_in = 1
        num_channels_out = 3
    else:
        raise ValueError('data must be either cross_GB, bubbles, gelslim, or cross_BG')
    
    if args.grayscale:
        num_channels_in = 1
        num_channels_out = 1

    if args.combined:
        num_channels_in = 3
        num_channels_out = 3

    model = model_definition(args.model_type, num_channels_in, num_channels_out, args.hidden_size, args.k, args.device, mod = args.mod, single=args.single)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initializing
    ncol = 8
    images = [train_imgs[0], train_imgs[1],
              center_val_imgs[0], center_val_imgs[1],
              tool_val_imgs[0], tool_val_imgs[1]]
    captions = ["Training Inputs", "Training Ground Truth", "Center Generalization Inputs", "Center Generalization Ground Truth", "Tool Generalization Inputs", "Tool Generalization Ground Truth"]
    logging_image_grid(images, captions, step = 0, ncol=ncol, normalize = True)

    # Training
    for epoch in range(args.num_epochs):
        print(f"Epoch = {epoch}/{args.num_epochs}")
        train_recons, train_vq = model.train_model(iter(train_loader), model, optimizer, args)
        center_val_recons, center_val_vq = model.val(iter(val_loader), model, args)

        with open('{0}/best.pt'.format(save_filename), 'wb') as f:
            torch.save(model.state_dict(), f)

        if epoch % logging_freq == 0:
            tools_val_recons = 0
            tools_val_vq = 0
            for i, tool in enumerate(tool_val_names):
                tool_val_recons, tool_val_vq = model.val(iter(DataLoader(tool_val_datasets[i], batch_size=args.batch_size, shuffle=False)), model, args)
                tools_val_recons += tool_val_recons
                tools_val_vq += tool_val_vq   
                wandb.log(data = {'Tool ('+ tool + ') Reconstruction Loss': tool_val_recons, 'Tool ('+ tool + ') Second Loss': tool_val_vq}, step = epoch)
            
            tools_val_recons /= len(tool_val_names)
            tools_val_vq /= len(tool_val_names)
            wandb.log(data = {'Training Reconstruction Loss': train_recons, 'Training Second Loss': train_vq, 'Center Generalization Reconstruction': center_val_recons, 'Center Generalization Second Loss': center_val_vq, 'Tool Generalization Reconstruction Loss': tools_val_recons, 'Tool Generalization Second Loss': tools_val_vq}, step = epoch)

            # Loading samples
            train_outputs = model.generate_visualization_samples(train_imgs[0], model, args.device)
            center_val_outputs = model.generate_visualization_samples(center_val_imgs[0], model, args.device)
            tool_val_outputs = model.generate_visualization_samples(tool_val_imgs[0], model, args.device)
            
            images = [train_outputs, 
                      center_val_outputs, 
                      tool_val_outputs]
            
            captions = ["Training Outputs", "Center Generalization Outputs", "Tool Generalization Outputs"]
            logging_image_grid(images, captions, step = epoch, ncol=ncol, normalize = True)
        

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # Latent space
    hidden_size_default = 256
    k_default = 16385
    parser.add_argument('--hidden-size', type=int, default=hidden_size_default,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=k_default,
        help='number of latent vectors (default: 16385)')

    # Optimization
    batch_size_default = 32
    num_epochs_default = 2500
    lr_default = 2e-4
    beta_default = 1.0
    parser.add_argument('--batch-size', type=int, default=batch_size_default,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=num_epochs_default,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=lr_default,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=beta_default,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--model_type', action='store')
    parser.add_argument('--output-folder', type=str, default='model',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--debug', action='store_true')

    # Dataset selection
    parser.add_argument('--filtered_data', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--data', type=str, default='cross')
    parser.add_argument('--dataset', type=str, default='new')
    parser.add_argument('--mod', type=str, default='1')
    parser.add_argument('--gel_distortion', action='store_false')
    parser.add_argument('--random_sensor', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--flipping', action='store_true')

    args = parser.parse_args()

    if args.random_sensor:
        args.single = True

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')

    folder = args.model_type + '_dataset_' + args.dataset + '_data_' + args.data + '_mod_' + args.mod 
    
    if args.filtered_data:
        folder += '_filtered_data'
    
    if args.combined:
        folder += '_combined'

    if args.single:
        folder += '_single'
    
    if args.grayscale:
        folder += '_grayscale'

    if args.cropped:
        folder += '_cropped'
    
    if not args.gel_distortion:
        folder += '_NGD'
    
    if args.random_sensor:
        folder += '_random_sensor'
    
    if args.color_jitter:
        folder += '_color_jitter'
    
    if args.rotation:
        folder += '_rotation'
    
    if args.flipping:
        folder += '_flipping'

    if not(args.hidden_size == hidden_size_default):
        folder += '_D' + str(args.hidden_size)

    if not(args.k == k_default):
        folder += '_K' + str(args.k)

    if not(args.batch_size == batch_size_default):
        folder += '_B' + str(args.batch_size)

    if not(args.num_epochs == num_epochs_default):
        folder += '_E' + str(args.num_epochs)

    if not(args.lr == lr_default):
        folder += '_LR' + str(args.lr)

    if not(args.beta == beta_default):
        folder += '_BETA' + str(args.beta)

    args.output_folder = folder + '_run_' + args.output_folder
    
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)