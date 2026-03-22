import os
import time
import argparse
import random as rn

import numpy as np
import torch

from dataset import dataset_h5
from model import SSANet
from trainOps import awgn
from utils import loadTxt, lmat, psnr, sam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_list', default='./test_path/val_vis_1.txt')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint .pth')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--bands', type=int, default=172, help='Base bands used to build the model')
    parser.add_argument('--cr', type=int, default=1)

    parser.add_argument('--val_hr', type=int, default=256)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--width', type=int, default=4)

    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--snr', type=float, default=99999)

    # variable band evaluation
    parser.add_argument('--missing_ratio', type=float, default=0.0, help='Drop this ratio of bands (0 means full bands)')
    parser.add_argument('--input_bands', type=int, default=None, help='If set, use exactly this many bands (overrides missing_ratio)')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


from typing import Optional


def _pick_band_indices(total_bands: int, missing_ratio: float, input_bands: Optional[int], seed: int) -> Optional[np.ndarray]:
    if input_bands is not None:
        keep = int(input_bands)
        if keep <= 0 or keep > total_bands:
            raise ValueError(f'input_bands must be in [1,{total_bands}], got {keep}')
    else:
        if missing_ratio <= 0:
            return None
        keep = max(1, int(round(total_bands * (1.0 - missing_ratio))))

    rng = np.random.default_rng(seed)
    idx = rng.choice(total_bands, size=keep, replace=False)
    idx.sort()
    return idx


def main() -> None:
    args = parse_args()

    device = args.device

    val_files = loadTxt(args.val_list)
    val_loader = torch.utils.data.DataLoader(
        dataset_h5(val_files, mode='Validation', root='', width=args.width, expected_bands=args.bands),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # Build model (base bands fixed)
    model = SSANet(cr=args.cr, bands=args.bands, variable_bands=True)

    state_dict = torch.load(args.ckpt, map_location=device)

    if device == 'cpu':
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    band_idx = _pick_band_indices(args.bands, args.missing_ratio, args.input_bands, args.seed)
    band_ids = None
    if band_idx is not None:
        band_ids = torch.from_numpy(band_idx).long().to(device)

    rmses, sams, psnrs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for _, (vx, vfn) in enumerate(val_loader):
            vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
            vx = vx.to(device).permute(0, 3, 1, 2).float()  # [N, bands, H, W]

            if band_ids is not None:
                vx = vx.index_select(1, band_ids)

            if args.sigma > 0:
                encoded, _ = model(vx, mode=1, band_ids=band_ids)
                base_out = model(awgn(encoded, args.snr), mode=2)
                if band_ids is not None:
                    base_out = model.module.band_mapper.from_base(base_out, band_ids=band_ids, out_bands=vx.shape[1])
                val_dec = base_out
            else:
                val_dec, _, _ = model(vx, band_ids=band_ids)

            val_batch_size = len(vfn)
            out_bands = val_dec.shape[1]
            img = [np.zeros((args.val_hr, args.val_hr, out_bands)) for _ in range(val_batch_size)]
            val_dec = val_dec.permute(0, 2, 3, 1).cpu().numpy()

            cnt = 0
            for bt in range(val_batch_size):
                for z in range(0, args.val_hr, args.interval):
                    img[bt][:, z : z + args.width, :] = val_dec[cnt]
                    cnt += 1

                gt = lmat(vfn[bt]).astype(np.float64)
                if band_idx is not None:
                    gt = gt[:, :, band_idx]

                # 去除 GT 中的零值（沿用原实现）
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        k = 0
                        while gt[i][j + k].sum() == 0:
                            k += 1
                        gt[i][j] = gt[i][j + k]

                maxv, minv = np.max(gt), np.min(gt)
                img[bt] = img[bt] * (maxv - minv) + minv

                sams.append(sam(gt, img[bt]))
                psnrs.append(psnr(gt, img[bt]))
                rmses.append(np.sqrt(np.mean((gt - img[bt]) ** 2)))

                print('{:25} '.format('/'.join(vfn[bt].split('/')[-2:])) + ' PSNR: %.3f RMSE: %.3f SAM: %.3f' % (psnrs[-1], rmses[-1], sams[-1]))

    avg_time = (time.time() - start_time) / max(1, len(sams))
    print('\nval-RMSE: %.3f, val-SAM: %.3f, PSNR:%.3f, AVG-Time: %.3f , CR: %.f, SNR: %.f' % (np.mean(rmses), np.mean(sams), np.mean(psnrs), avg_time, args.cr, args.snr))


if __name__ == '__main__':
    main()
