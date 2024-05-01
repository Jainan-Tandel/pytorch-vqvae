import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import tqdm
from modules import VectorQuantizedVAE, to_scalar
from datasets import ISIC
import wandb
import datetime


def train(data_loader, model, optimizer, args, batch_bar=None):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        wandb.log({'loss/train/reconstruction': loss_recons.item(), 'loss/train/quantization': loss_vq.item()}, args.steps)
        optimizer.step()
        args.steps += 1
        if batch_bar:
            batch_bar.update()

def test(data_loader, model, args, batch_bar=None):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)
            if batch_bar:
                batch_bar.update()


        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
        
    wandb.log({'loss/test/reconstruction': loss_recons.item(),'loss/test/quantization': loss_vq.item() }, step=args.steps)
    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    wandb.init(project = "VQVAE_DL",
               entity = "m23csa010",
               config={
                   "lr":args.lr,
                   "epochs":args.num_epochs,
                   "latent_size":args.hidden_size,
                   "latent_number":args.k,
                   "input_crop_size":args.input_crop_size,
                   "input_resize_size":args.input_resize_size,
                   "batch_size":args.batch_size,
                   "dataset":args.dataset,
                   "device":args.device
               },
               name=f"VQVAE_{timestamp}",
               dir='./logs/{0}'.format(args.output_folder),
               )
    
    save_filename = './models/{0}'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset

    elif args.dataset == 'isic':
        transform = transforms.Compose([
            transforms.CenterCrop(size=(args.input_crop_size,args.input_crop_size)),
            transforms.Resize(size=(args.input_resize_size,args.input_resize_size)),
            # transforms.RandomResizedCrop(size = (64,64), scale = (1,1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_dataset = ISIC(args.data_folder, train=True, transform=transform)
        valid_dataset = ISIC(args.data_folder, valid=True, transform=transform)
        test_dataset = ISIC(args.data_folder, test=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = -1.
    batch_bar = tqdm.tqdm(range(len(train_loader)),desc="Training")
    for epoch in tqdm.tqdm(range(args.num_epochs), desc="Epochs"):
        batch_bar.reset(total=(len(train_loader)))
        batch_bar.set_description(desc="Training")
        train(train_loader, model, optimizer, args, batch_bar)
        batch_bar.reset(total=(len(valid_loader)))
        batch_bar.set_description(desc="Validating")
        loss, _ = test(valid_loader, model, args, batch_bar)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)
    batch_bar.close()

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default='/data',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='isic',
        help='name of the dataset (mnist, fashion-mnist, cifar10, isic)')
    parser.add_argument('--output-folder', type=str, default='models/vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=1024,
        help='number of latent vectors (default: 1024)')
    
    # Image preprocess
    parser.add_argument('--input-crop-size', type=int,default=448,
        help='size of the cropped input image (default: 448)')
    parser.add_argument('--input-resize-size', type=int,default=128,
        help='size of the cropped input image (default: 128)')
    
    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    args = parser.parse_args()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))

    args.steps = 0
    main(args)
