import os
import torch
from touch2touch.vq_vae.training import main

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