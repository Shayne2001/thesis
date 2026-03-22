import numpy as np
import torch, os
import random as rn
from typing import Optional
from torch.utils import data
from scipy.io import loadmat


_AVIRIS_224_TO_172_DROP_INDEX = list(range(0, 10)) + list(range(103, 116)) + list(range(151, 170)) + list(range(214, 224))


class dataset_h5(torch.utils.data.Dataset):
    def __init__(
        self,
        X,
        img_size=256,
        crop_size=128,
        width=4,
        root='',
        mode='Train',
        marginal=40,
        expected_bands: Optional[int] = 172,
    ):
        super(dataset_h5, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))

        self.mode = mode
        self.crop_size = crop_size
        self.img_size = img_size
        self.marginal = marginal
        self.width = width
        self.expected_bands = expected_bands

    def _maybe_reduce_bands(self, x: np.ndarray, fn: str) -> np.ndarray:
        """把 AVIRIS 的 224 bands 处理成常用的 172 bands。

        只有当 expected_bands == 172 时才会触发该裁剪；否则保持原 band 数不变。
        """

        if self.expected_bands == 172 and x.shape[2] == 224:
            x = np.delete(x, _AVIRIS_224_TO_172_DROP_INDEX, axis=2)

        if self.expected_bands is not None and x.shape[2] != self.expected_bands:
            raise ValueError(
                f"File {fn} has {x.shape[2]} bands, but expected_bands={self.expected_bands}. "
                f"(If you want to keep original bands, pass expected_bands=None.)"
            )

        return x

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.fns[index])

        x = loadmat(fn)
        x = x[list(x.keys())[-1]].astype(np.float32)

        xmin = np.min(x)
        xmax = np.max(x)

        x = self._maybe_reduce_bands(x, fn)

        if self.mode == 'Train':
            shifting_h = (self.img_size - self.crop_size) // 2
            shifting_w = (self.img_size - self.width) // 2 - self.marginal
            xim, yim = rn.randint(0, shifting_w), rn.randint(0, shifting_h)
            h = yim + self.crop_size

            xx = []
            for k in range(self.marginal):
                y = x[yim:h, xim + k : xim + self.width + k, :]

                if rn.random() > 0.5:
                    y = y[::-1, :, :]
                if rn.random() > 0.5:
                    y = y[:, ::-1, :]

                xx.append(torch.from_numpy(y.copy()))

            x = torch.stack(xx)
        else:
            xx = []
            for k in range(0, x.shape[0], self.width):
                y = x[:, k : k + self.width, :]
                xx.append(torch.from_numpy(y.copy()))
            x = torch.stack(xx)

        if xmin == xmax:
            # 避免出现 NaN；保持返回形状与当前 mode 一致
            return torch.zeros_like(x), fn

        x = (x - xmin) / (xmax - xmin)
        return x, fn

    def __len__(self):
        return self.n_images
