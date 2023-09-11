# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:01:58 2020

@author: ZJU
"""

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),  #256
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/data/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/data/WikiArt/train',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=6.0)
parser.add_argument('--style_local_weight', type=float, default=0.5)
parser.add_argument('--content_weight', type=float, default=5.0)
parser.add_argument('--attention_weight', type=float, default=50000.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()}], lr=args.lr)

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    img, loss_c, loss_s, loss_s_local, loss_attention, l_identity1, l_identity2 = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_s_local = args.style_local_weight * loss_s_local
    loss_attention = args.attention_weight * loss_attention
    loss = loss_c + loss_s + loss_s_local + loss_attention + l_identity1 * 50 + l_identity2 * 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_style_local', loss_s_local.item(), i + 1)
    writer.add_scalar('loss_attention', loss_attention.item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i == 0) or ((i + 1) % 500 == 0):  # 1000
        output = torch.cat([style_images, content_images, img], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        #save_image(output, str(output_name), args.batch_size)
        save_image(output, str(output_name))
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
writer.close()