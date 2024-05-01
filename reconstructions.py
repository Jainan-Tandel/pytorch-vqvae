import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import tqdm
from modules import VectorQuantizedVAE, to_scalar, GatedPixelCNN
from datasets import MiniImagenet, ISIC
import matplotlib.pyplot as plt
import datetime
from torchvision.models import inception_v3, Inception_V3_Weights

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def generate_pixel_samples(labels,vae,prior, args, shape=(16,16)):
    with torch.no_grad():
        pixelout = prior.generate(labels, batch_size=labels.shape[0], shape=shape)
        x_tilde = vae.decode(pixelout)
    return x_tilde

def unnormalise(images, mean, std):
    transform = transforms.Compose([
        transforms.Normalize(mean=(0,0,0),std = [1/x for x in std]),
        transforms.Normalize(mean= [-x for x in mean], std=1)]
    )
    images = transform(images)
    return images

def plot_and_calculate_inception_score(test_loader, model, prior, prior_shape, unnormalize_params, num_images=100):
    inception_model = inception_v3(weights = Inception_V3_Weights.DEFAULT, transform_input=False).eval().to(args.device)

    preds_list = []
    print(num_images//16 + min((num_images%16),1))
    for i, (_, fixed_labels) in tqdm.tqdm(enumerate(test_loader),desc="Generation", total = num_images//16 + min((num_images%16),1)):
        with torch.no_grad():
            artificial = generate_pixel_samples(fixed_labels, model, prior, args, shape = prior_shape)

            batch_preds = F.softmax(inception_model(artificial.to(args.device)), dim=1)

            preds_list.append(batch_preds.cpu().numpy())

            artificial = unnormalise(artificial, **unnormalize_params)

            fig,ax = plt.subplots(nrows=4,ncols=4)
            for idx in range(min(len(artificial),16)):
                ax[idx//4][idx%4].imshow(artificial[idx].permute(1,2,0), vmin=0,vmax=1)
                ax[idx//4][idx%4].axis('off')
            
            plt.title(f"Fake Batch {i}")
            plt.savefig(os.path.join(args.output_folder,f"fake_figure_{i}.png"))
            plt.close()
        if i *16>num_images: break

    preds = torch.tensor(np.concatenate(preds_list, axis=0))
    mean_preds = torch.mean(preds, dim=0)
    std_preds = torch.std(preds, dim=0)

    kl_divs = F.kl_div(mean_preds.log(), std_preds, reduction='sum')
    avg_kl_div = torch.mean(kl_divs)
    inception_score = torch.exp(avg_kl_div)

    return inception_score.item()


def main(args):
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
        unnormalize_params = {'mean':(0.5, 0.5, 0.5),'std':(0.5, 0.5, 0.5)}
        valid_dataset = test_dataset
 
    elif args.dataset == 'isic':
        transform = transforms.Compose([
            transforms.CenterCrop(size=(args.input_crop_size,args.input_crop_size)),
            transforms.Resize(size=(args.input_resize_size,args.input_resize_size)),
            # transforms.RandomResizedCrop(size = (64,64), scale = (1,1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        # train_dataset = ISIC(args.data_folder, train=True,
        #     download=True, transform=transform)
        # valid_dataset = ISIC(args.data_folder, valid=True,
        #     download=True, transform=transform)
        test_dataset = ISIC(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3
        args.num_classes = None
        unnormalize_params = {'mean':(0.485, 0.456, 0.406),'std':(0.229, 0.224, 0.225)}

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #     batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=False)

    fixed_images, _ = next(iter(test_loader))

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)

    weights = torch.load('models/models/vqvae/best.pt')
    model.load_state_dict(weights)
    prior_shape = model.get_shape(fixed_images)[2:]

    prior = GatedPixelCNN(args.k, args.hidden_size_prior, args.num_layers, n_classes = args.num_classes ).to(args.device)#len(train_dataset._label_encoder)).to(args.device)
    
    pixelweights = torch.load('models/models/pixelcnn_prior/prior.pt')
    prior.load_state_dict(pixelweights)
    model.eval()
    prior.eval()
    
    for i, (fixed_images, _) in tqdm.tqdm(enumerate(test_loader),desc="Reconstruction", total=len(test_loader)):
        with torch.no_grad():
            reconstruction = generate_samples(fixed_images, model, args)
            # artificial = generate_pixel_samples(fixed_labels, model, prior, args, shape = prior_shape)

            fixed_images = unnormalise(fixed_images, **unnormalize_params)
            reconstruction = unnormalise(reconstruction, **unnormalize_params)
            # artificial = unnormalise(artificial, **unnormalize_params)

            # flim = (torch.min(fixed_images), torch.max(fixed_images))
            # rlim = (torch.min(reconstruction), torch.max(reconstruction))
            # alim = (torch.min(artificial), torch.max(artificial))

            fig,ax = plt.subplots(nrows=4,ncols=8)
            for idx in range(min(len(fixed_images),16)):
                ax[idx//8][idx%8].imshow(fixed_images[idx].permute(1,2,0), vmin=0,vmax=1)
                ax[idx//8][idx%8].axis('off')
            
            for idx in range(16, min(len(fixed_images),16) +16):
                ax[idx//8][idx%8].imshow(reconstruction[idx-16].permute(1,2,0),vmin=0,vmax=1)
                ax[idx//8][idx%8].axis('off')

            # for idx in range(32,48):
            #     ax[idx//8][idx%8].imshow(artificial[idx-32].permute(1,2,0),vmin=0,vmax=1)
            #     ax[idx//8][idx%8].axis('off')

            fig.suptitle(f"Reconstruction batch = {i}")

            plt.savefig(os.path.join(args.output_folder,f"test_figure_{i}.png"))
            plt.close()
    
    Inception_score = plot_and_calculate_inception_score(test_loader, model, prior, prior_shape, unnormalize_params)

    print(Inception_score)

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reconstructor for VQ-VAE and PixelCNN generation')

    # General
    parser.add_argument('--data-folder', type=str, default='/data',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str, default='isic',
        help='name of the dataset (mnist, fashion-mnist, cifar10, isic)')
    parser.add_argument('--output-folder', type=str, default=f'figures/{timestamp}',
        help='name of the output folder (default: figures/\'timestamp\')')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')
    
    # Image preprocess
    parser.add_argument('--input-crop-size', type=int,default=448,
        help='size of the cropped input image (default: 448)')
    parser.add_argument('--input-resize-size', type=int,default=128,
        help='size of the cropped input image (default: 128)')
    
    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=1024,
        help='number of latent vectors (default: 1024)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=10,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Prior
    parser.add_argument('--hidden-size-prior', type=int, default=80,
        help='hidden size for the PixelCNN prior (default: 80)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes of data')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./{0}'.format(args.output_folder)):
        os.makedirs('./{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
