import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from glob import glob
import net


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def test_transform2():
    transform_list = [transforms.Resize(512)]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def TSSAT(cf, sf, patch_size=5, stride=1):  # cf,sf  Batch_size x C x H x W
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64
    #print(cf.size())
    kh, kw = patch_size, patch_size
    sh, sw = stride, stride

    # Create convolutional filters by style features
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = sf_unfold.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(b, -1, c, kh, kw)
    patches_norm = torch.norm(patches.reshape(*patches.shape[:2], -1), dim=2).reshape(b, -1, 1, 1, 1)
    patches_norm = patches / patches_norm
    # patches size is 2 x 3844 x 256 x 3 x 3

    cf = adaptive_instance_normalization(cf, sf)
    for i in range(b):
        cf_temp = cf[i].unsqueeze(0)  # [1 x 256 x 64 x 64]
        patches_norm_temp = patches_norm[i]  # [3844, 256, 3, 3]
        patches_temp = patches[i]

        _, _, ch, cw = cf.size()
        for c_i in range(0, ch, patch_size):
            ###################################################
            if (c_i + patch_size) > ch:
                break
            elif (c_i + 2*patch_size) > ch:
                ckh = ch - c_i
            else:
                ckh = patch_size
            ###################################################

            for c_j in range(0, cw, patch_size):
                ###################################################
                if (c_j + patch_size) > cw:
                    break
                elif (c_j + 2 * patch_size) > cw:
                    ckw = cw - c_j
                else:
                    ckw = patch_size
                ###################################################

                temp = cf_temp[:, :, c_i:c_i + ckh, c_j:c_j + ckw]
                conv_out = F.conv2d(temp, patches_norm_temp, stride=patch_size)
                index = conv_out.argmax(dim=1).squeeze()
                style_temp = patches_temp[index].unsqueeze(0)
                stylized_part = adaptive_instance_normalization(temp, style_temp)

                if c_j == 0:
                    p = stylized_part
                else:
                    p = torch.cat([p, stylized_part], 3)

            if c_i == 0:
                q = p
            else:
                q = torch.cat([q, p], 2)

        if i == 0:
            out = q
        else:
            out = torch.cat([out, q], 0)

    return out


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = 'content/1.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = 'style/1.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = 'output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args('')

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform2()

#############################################################################

content = content_tf(Image.open(args.content))
style = style_tf(Image.open(args.style))

content = content.to(device).unsqueeze(0)
style = style.to(device).unsqueeze(0)

with torch.no_grad():
    for x in range(args.steps):

        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        content = decoder(TSSAT(Content4_1, Style4_1))

        content.clamp(0, 255)

    content = content.cpu()

    output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
        args.output, splitext(basename(args.content))[0],
        splitext(basename(args.style))[0], args.save_ext
    )

    save_image(content, output_name)
