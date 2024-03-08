import cv2
import numpy as np
import torch


def picture_load(img_path):

    to_tensor = ToTensor(
        mean=0.0558,  # city, gray
        std=0.1208,
    )

    # 从args.img_path读取路径打开图片返回（高度，宽度，通道数）的元组。 (1024, 2048, 3)   ::-1 表示opencv的读取BGR -> RGB
    im = cv2.imread(img_path,0)

    im = (dict(im=im, lb=im))
    im = im['im']
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)
    im = im.unsqueeze(0)

    return im



class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def  __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)