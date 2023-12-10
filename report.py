import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt, visualize_depth
import metrics

from dataset import KlevrDataset
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='./dataset/',
                        help='root directory of dataset')
    parser.add_argument('--folder_output', type=str, default='bash_output/report/',
                        help='output folder name')
    parser.add_argument('--split', type=str, default='val',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=True, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=True, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        test_time=True,
                        white_back=False)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    dir_name = args.folder_output
    kwargs = {'root_dir': args.root_dir,}
              #'split': args.split,
              #'img_wh': tuple(args.img_wh)}

    dataset = KlevrDataset
    dataset = dataset(split='val', **kwargs)
    
    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    ssims = []
    lpips = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        name = sample['name']
        
        if args.save_depth:
            #plt.subplots(figsize=(8, 8))
            plt.axis('off')
            plt.tight_layout()
            depth = results['depth_fine'].view(h, w)
            #depth = visualize_depth(depth).permute(1,2,0)
            #print(depth)
            #print(depth.shape)
            #cv2.imwrite(os.path.join(dir_name, f'depth_{name:05d}.png'), depth)
            #plt.title(f'depth_{name:05d}')
            plt.imshow(visualize_depth(depth).permute(1,2,0))
            plt.savefig(os.path.join(dir_name, f'depth_{name:05d}.png'))
            
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{name:05d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
            ssims += [metrics.rgb_ssim(img_pred, img_gt.cpu().numpy(), max_val=1).item()]
            #ssims += [metrics.rgb_ssim(img_gt.cpu().numpy(), img_pred, max_val=1).item()]
            c2w = sample['c2w'].cuda()
            c2w = torch.Tensor(c2w)
            lpips += [metrics.rgb_lpips(img_gt.cpu().numpy(), img_pred, net_name='vgg', device=c2w.device)]
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        mean_lpips = np.mean(lpips)
        print(f'Mean PSNR : {mean_psnr}')
        print(f'Mean SSIM : {mean_ssim}')
        print(f'Mean LPIPS : {mean_lpips}')