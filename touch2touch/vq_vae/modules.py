import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
# from simclr import SimCLR
# from simclr.modules import get_resnet
from touch2touch.vq_vae.functions import vq, vq_st
import numpy as np
# from barlow import BarlowTwins
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        #import pdb; pdb.set_trace()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            # nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, output_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            # nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        #import pdb; pdb.set_trace()
        return x_tilde, z_e_x, z_q_x
    
class VectorQuantizedVAEResnetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dim, K=512, single=False):
        super().__init__()
        self.single = single
        resnet_encoder = torchvision.models.resnet50(pretrained=False)
        self.encoder = (torch.nn.Sequential(*(list(resnet_encoder.children())[:-2])))
        self.encoder.add_module('last_layer', nn.Conv2d(2048, dim, 1, 1, 1))

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            # START ADDING for output shape match
            nn.ConvTranspose2d(dim, dim, 1, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # END ADDING for output shape match
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            # nn.Tanh()
        )

        self.criterion = nn.MSELoss()

        # self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde
    
    def train_model(self, data_loader, model, optimizer, args):
        model.train()
        loss_recons_b, loss_vq_b, loss_kl_b = 0., 0., 0.
        for images, labels, _ in data_loader:
            if self.single:
                images = images[:,1]
                labels = labels[:,1]
            else:
                images = torch.cat([images[:,0], images[:,1]], dim = 0)
                labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

            optimizer.zero_grad()
            x_tilde, z_e_x, z_q_x = model(images)
            # x_tilde, kl_div = model(images)

            # Have to add projection to 3D point cloud
            # open3d.geometry.create_point_cloud_from_depth_image(depth, intrinsic, extrinsic=(with default value), depth_scale=1000.0, depth_trunc=1000.0, stride=1)

            # # Reconstruction loss
            # # loss_recons = F.mse_loss(x_tilde, labels)
            # loss_recons = 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)
            # # Vector quantization objective
            # loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # # Commitment objective
            # loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            # loss_recons_b += loss_recons
            # loss_vq_b += loss_vq

            # loss = loss_recons + loss_vq + args.beta * loss_commit
            # # loss = loss_recons + kl_div

            loss_emb = torch.mean((z_q_x.detach()-z_e_x)**2) + 0.25 * \
            torch.mean((z_q_x - z_e_x.detach()) ** 2)   # Take beta as 0.25
        
            # Reconstruction loss
            mse_loss = self.criterion(x_tilde, images) / torch.var(images/255.0)  # Autoencoder, divide by variance
            loss=loss_emb+mse_loss
            loss_recons_b += mse_loss
            loss_vq_b += loss_emb
            loss.backward()

            optimizer.step()
            args.steps += 1

        return loss_recons_b.item()/len(data_loader), loss_vq_b.item()/len(data_loader)

    def val(self, data_loader, model, args):
        model.eval()
        with torch.no_grad():
            loss_recons, loss_vq, loss_kl = 0., 0., 0.
            for images, labels, _ in data_loader:
                if self.single:
                    images = images[:,1]
                    labels = labels[:,1]
                else:
                    images = torch.cat([images[:,0], images[:,1]], dim = 0)
                    labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

                x_tilde, z_e_x, z_q_x = model(images)
                # # x_tilde, kl_div = model(images)
                # loss_recons += 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)
                # # loss_recons += F.mse_loss(x_tilde, labels)
                # loss_vq += F.mse_loss(z_q_x, z_e_x)
                # # loss_kl += kl_div

                loss_emb = torch.mean((z_q_x.detach()-z_e_x)**2) + 0.25 * \
                torch.mean((z_q_x - z_e_x.detach()) ** 2)   # Take beta as 0.25
                # Reconstruction loss
                mse_loss = self.criterion(x_tilde, images) / torch.var(images/255.0)  # Autoencoder, divide by variance
                loss=loss_emb+mse_loss
                loss_recons += mse_loss
                loss_vq += loss_emb

            loss_recons /= len(data_loader)
            loss_vq /= len(data_loader)

        return loss_recons.item(), loss_vq.item()
    
    def generate_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            x_tilde, _, _ = model(images)
        return x_tilde
    
    def generate_visualization_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            if self.single:
                x_tilde, _, _ = model(images)
                output = x_tilde
            else:
                images_l, images_r= torch.split(images, int(images.shape[2]/2), dim=2)
                images = torch.cat([images_l, images_r], dim = 0)
                x_tilde, _, _ = model(images)
                x_tilde_l, x_tilde_r= torch.split(x_tilde, int(x_tilde.shape[0]/2), dim=0)
                output = torch.cat([x_tilde_l, x_tilde_r], dim = 2)
        return output

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, _ = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
    
class VAEResnetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dim, single=False):
        super().__init__()
        self.single = single
        resnet_encoder = torchvision.models.resnet50(pretrained=False)
        self.encoder = (torch.nn.Sequential(*(list(resnet_encoder.children())[:-2])))
        self.fc_mu = nn.Linear(2048*4*4, dim)
        self.fc_var = nn.Linear(2048*4*4, dim)
        self.dim = dim

        self.decoder_input = nn.Linear(dim, dim*6*6)
        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            # START ADDING for output shape match
            nn.ConvTranspose2d(dim, dim, 1, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # END ADDING for output shape match
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            # nn.Tanh()
        )

        # self.small_decoder_input = nn.Linear(dim, dim*6*6)
        # self.small_decoder = nn.Sequential(

        # self.apply(weights_init)
    
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, dim, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # import pdb; pdb.set_trace()
        result = result.view(-1, dim,6,6)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def generate_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            x_tilde,_,_,_ = model(images)
        return x_tilde
    
    def generate_visualization_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            if self.single:
                x_tilde, _, _, _ = model(images)
                output = x_tilde
            else:
                images_l, images_r= torch.split(images, int(images.shape[2]/2), dim=2)
                images = torch.cat([images_l, images_r], dim = 0)
                x_tilde, _, _, _ = model(images)
                x_tilde_l, x_tilde_r= torch.split(x_tilde, int(x_tilde.shape[0]/2), dim=0)
                output = torch.cat([x_tilde_l, x_tilde_r], dim = 2)
        return output
    
    def train_model(self, data_loader, model, optimizer, args):
        loss_recons_b, loss_vq_b, loss_kl_b = 0., 0., 0.
        model.train()
        for images, labels, _ in data_loader:
            if self.single:
                    images = images[:,1]
                    labels = labels[:,1]
            else:
                images = torch.cat([images[:,0], images[:,1]], dim = 0)
                labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

            optimizer.zero_grad()
            x_tilde, input, mu, log_var = model(images)
            loss_recons = F.mse_loss(x_tilde, labels)
            # loss_recons = 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)
            # Vector quantization objective
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            

            loss_recons_b += loss_recons
            loss_kl_b += kld_loss

            loss = loss_recons + 0.00025 * kld_loss
            # loss = loss_recons + kl_div
            loss.backward()

            optimizer.step()
            args.steps += 1

        return loss_recons_b.item()/len(data_loader), loss_kl_b.item()/len(data_loader)
    
    def val(self, data_loader, model, args):
        model.eval()
        with torch.no_grad():
            loss_recons, loss_vq, loss_kl = 0., 0., 0.
            for images, labels, _ in data_loader:
                if self.single:
                    images = images[:,1]
                    labels = labels[:,1]
                else:
                    images = torch.cat([images[:,0], images[:,1]], dim = 0)
                    labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

                x_tilde, input, mu, log_var = model(images)
                # x_tilde, kl_div = model(images)
                # loss_recons += 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)
                loss_recons += F.mse_loss(x_tilde, labels)
                loss_kl += torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                # loss_kl += kl_div

            loss_recons /= len(data_loader)
            loss_vq /= len(data_loader)

        return loss_recons.item(), loss_kl.item()

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(self.dim, z), input, mu, log_var
    

class SimCLR_Style_Transfer(nn.Module):   # Need to change this as well
    def __init__(self, encoder, projector, decoder, encoded_shape):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.encoded_shape = encoded_shape

    def forward(self, x):
        '''
        x: gelslim input
        D: dim of input channels for decoder
        H: dim of height of decoder input
        W: dim of width of decoder input 
        ''' 
        D, H, W = self.encoded_shape
        with torch.no_grad():
            latent = self.encoder(x)
        projection = self.projector(latent).view(-1, D, H, W)
        bubbles_img = self.decoder(projection)

        return bubbles_img
    def generate_samples(self, images, model, device):
        with torch.no_grad():
            images = images.to(device)
            x_tilde = model(images)
        return x_tilde
    def generate_visualization_samples(self, images, model, device):
        with torch.no_grad():
            images_l, images_r= torch.split(images, int(images.shape[2]/2), dim=2)
            images = torch.cat([images_l, images_r], dim = 0)
            x_tilde = model(images)
            x_tilde_l, x_tilde_r= torch.split(x_tilde, int(x_tilde.shape[0]/2), dim=0)
            output = torch.cat([x_tilde_l, x_tilde_r], dim = 2)
        return output
    
def model_definition(model_type, num_channels_in, num_channels_out, hidden_size, k, device, single=False, mod='1'):   # Add new parameter pretrained weight
    if model_type == 'VAE':
        model = VAEResnetEncoder(num_channels_in, num_channels_out, hidden_size, single=single).to(device)
    elif model_type == 'VQ-VAE':
        model = VectorQuantizedVAEResnetEncoder(num_channels_in, num_channels_out, hidden_size, k, single=single).to(device)
    elif model_type == 'VQ-VAE-small':
        model = VectorQuantizedVAE_standalone(num_channels_in, num_channels_out, hidden_size, k, mod=mod, single=single).to(device)
    elif model_type == 'VQ-VAE-ViT':
        input_size=128
        input_sample = torch.ones([1, num_channels_in, input_size, input_size]).to(device)

        '''My vq-vae is only slightly different, has 1 output instead of 3, could also make that change in your own code'''
        vq_vae = VectorQuantizedVAE_ajitesh(num_channels_in, num_channels_out, hidden_size, k).to(device)  

        (B,D,H,W) = vq_vae.encoder(input_sample).shape
        encoded_shape = (D, H, W)
        print(encoded_shape)
        
        '''Change code to include this path as parameter, and for this time just change path'''
        path_to_vqvae_weights = '/home/samanta/tactile_style_transfer/tactile_style_transfer/scripts/working_model/vq_vae_reconstruction_gelslim.pt'
        encoder_pretrained_weigths = torch.load(path_to_vqvae_weights, map_location=torch.device(device))
        new_wts=encoder_pretrained_weigths.copy()
        for key in new_wts:
            if key.split('.')[0]=='decoder':
                del encoder_pretrained_weigths[key]

        gelslim_encoder=VectorQuantizedVAE_ajitesh(num_channels_in, num_channels_out, hidden_size, k).to(device)
        vq_vae_2 = VectorQuantizedVAE_ajitesh(num_channels_in, 1, hidden_size, k).to(device)
        bubbles_decoder = vq_vae_2.decoder.to(device)

        state_dict = gelslim_encoder.state_dict()
        encoder_keys = list(state_dict.keys())
        for i, key in enumerate(encoder_pretrained_weigths.keys()):
            state_dict[encoder_keys[i]] = encoder_pretrained_weigths[key]
        gelslim_encoder.load_state_dict(state_dict)

        projector=decoderViT(image_size = 32,
                                        channels=256,
                                        patch_size = 2,
                                        num_classes = 1000,
                                        dim = 1024,
                                        depth = 6,
                                        heads = 16,
                                        mlp_dim = 2048,
                                        dropout = 0.1,
                                        emb_dropout = 0.1).to(device)
        
        model = SimCLR_Style_Transfer(gelslim_encoder, projector, bubbles_decoder, encoded_shape)
            
        return model
    

    else:
        # Set Bubbles Decoder for training
        input_size = 128
        input_sample = torch.ones([1, num_channels_in, input_size, input_size]).to(device)
        vq_vae = VectorQuantizedVAE(num_channels_in, num_channels_out, hidden_size, k).to(device)
        (B,D,H,W) = vq_vae.encoder(input_sample).shape
        encoded_shape = (D, H, W)
        print(encoded_shape)
        bubbles_decoder = vq_vae.decoder
        
        # Set Gelslim Encoder, and projection layer for training
        resnet = "resnet50"
        projection_dim = D*H*W
        encoder = get_resnet(resnet, pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        simclr_model = SimCLR(encoder, projection_dim, n_features).to(device)
        gelslim_encoder = simclr_model.encoder
        projection_layers = simclr_model.projector
        
        # Set full model
        model = SimCLR_Style_Transfer(gelslim_encoder, projection_layers, bubbles_decoder, encoded_shape)
    # else:
    #     # Set Bubbles Decoder for training
    #     input_size = 128
    #     input_sample = torch.ones([1, num_channels_in, input_size, input_size]).to(device)
    #     vq_vae = VectorQuantizedVAE(num_channels_in, num_channels_out, hidden_size, k).to(device)
    #     (B,D,H,W) = vq_vae.encoder(input_sample).shape
    #     encoded_shape = (D, H, W)
    #     bubbles_decoder = vq_vae.decoder
        
    #     # Set Gelslim Encoder, and projection layer for training
    #     resnet = "resnet50"
    #     projection_dim = D*H*W
    #     encoder = get_resnet(resnet, pretrained=False)
    #     n_features = encoder.fc.in_features  # get dimensions of fc layer
    #     simclr_model = SimCLR(encoder, projection_dim, n_features).to(device)
    #     gelslim_encoder = simclr_model.encoder
    #     projection_layers = simclr_model.projector
        
    #     # Set full model
    #     model = SimCLR_Style_Transfer(gelslim_encoder, projection_layers, bubbles_decoder, encoded_shape)
    return model


'''New models added by Ajitesh'''
'''Functions used in the models, which are below these'''
'''1. For VQ-VAE'''
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)         # vq is not differentiable
        return latents

    def straight_through(self, z_e_x):   
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()   # Made 128x32x32x256
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())     # Straight through is differentiable
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, indices

'''2. For ViT, and for VQ-GAN Transformer'''
'''ViT Implementation'''

class VectorQuantizedVAE_ajitesh(nn.Module):
    def __init__(self, input_dim, output_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            # nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, index = self.codebook.straight_through(z_e_x)   # z_q_x_st is codes, added indices for trasformers
        x_tilde = self.decoder(z_q_x_st)
        return z_q_x_st # z_q_x_st added later only to get codes during transformer training, indices added for transformer
    

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class decoderViT(nn.Module):
    '''Changed it to use embeddings instead of images'''
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # print(img.size())    # 64x256x32x32
        x = self.to_patch_embedding(img)   # 64x1024x256
        b, n, _ = x.shape   
        
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        
        x = self.transformer(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            # nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


'''Models added'''
'''1. VQ-VAE Standalone'''
class VectorQuantizedVAE_standalone(nn.Module):
    def __init__(self, input_dim, output_dim, dim, K=512, mod='1', single=False):
        super().__init__()
        self.single = single
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, 4, 2, 1),
            # nn.Tanh()
        )

        self.apply(weights_init)
        self.criterion = nn.MSELoss()
        self.mod = mod

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, index = self.codebook.straight_through(z_e_x)   # z_q_x_st is codes, added indices for trasformers
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x  # z_q_x_st added later only to get codes during transformer training, indices added for transformer vqgan
    
    
    def train_model(self, data_loader, model, optimizer, args):
        loss_recons_b = 0.
        loss_emb_b = 0.
        for images, labels, _ in data_loader:
            optimizer.zero_grad()
            if not self.single:
                images = torch.cat([images[:,0], images[:,1]], dim = 0)
                labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

            x_tilde, z, z_q = model(images)

            # Embedding loss
            loss_emb = torch.mean((z_q.detach()-z)**2) + 0.25 * \
                torch.mean((z_q - z.detach()) ** 2)   # Take beta as 0.25
            
            # Reconstruction loss
            loss_recons = 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)    # Check if I used this or autoencoder one
            # loss_recons = self.criterion(x_tilde, images) / torch.var(images/255.0)  # Autoencoder, divide by variance
            tot_loss=loss_emb+loss_recons
            loss_recons_b += loss_recons
            loss_emb_b += loss_emb
            tot_loss.backward()

            optimizer.step()

        return loss_recons_b.item()/len(data_loader), loss_emb_b.item()/len(data_loader)

    def val(self, data_loader, model, args):
        model.eval()
        with torch.no_grad():
            loss_recons_b = 0.
            loss_emb_b = 0.
            for images, labels, _ in data_loader:
                if not self.single:
                    images = torch.cat([images[:,0], images[:,1]], dim = 0)
                    labels = torch.cat([labels[:,0], labels[:,1]], dim = 0)

                x_tilde, z, z_q = model(images)

                # Embedding loss
                loss_emb = torch.mean((z_q.detach()-z)**2) + 0.25 * \
                    torch.mean((z_q - z.detach()) ** 2)   # Take beta as 0.25
                
                # Reconstruction loss
                loss_recons = 0.5*(torch.log(2*np.pi*F.mse_loss(x_tilde, labels)) + 1)    # Check if I used this or autoencoder one

                tot_loss=loss_emb+loss_recons
                loss_recons_b += loss_recons
                loss_emb_b += loss_emb

        return loss_recons_b.item()/len(data_loader), loss_emb_b.item()/len(data_loader)

    def generate_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            if not self.single:
                images = torch.cat([images[:,0], images[:,1]], dim = 0)
            images = images.to(device)
            # import pdb; pdb.set_trace()
            x_tilde, _, _ = model(images)
        return x_tilde

    def generate_visualization_samples(self, images, model, device):
        model.eval()
        with torch.no_grad():
            if self.single:
                x_tilde, _, _ = model(images)
                output = x_tilde
            else:
                images_l, images_r= torch.split(images, int(images.shape[2]/2), dim=2)
                images = torch.cat([images_l, images_r], dim = 0)
                x_tilde, _, _ = model(images)
                x_tilde_l, x_tilde_r= torch.split(x_tilde, int(x_tilde.shape[0]/2), dim=0)
                output = torch.cat([x_tilde_l, x_tilde_r], dim = 2)      
        return output

'''1. VQ-VAE codes, used with Vision Transformer'''
