from os.path import splitext    
from os import listdir          
import numpy as np
from glob import glob           
import torch
from torch.utils.data import Dataset    
                                        
                                        
import logging                  
from PIL import Image           
from torchvision import transforms
import math
import torch.nn.functional as F


class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, mode='train', scale=1, mask_suffix='', transformer=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mode = mode
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transformer = transformer
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if not file.startswith('.')]    
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod    
                    
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)     
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)     # np.expand_dims: expand the shape of the array

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # tr = transforms.Normalize(mean=[0.5], std=[0.5])
        img_trans = torch.from_numpy(img_trans).type(torch.FloatTensor)
        # img_trans = tr(img_trans)
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.images_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')    

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        img = Image.open(img_file[0]).convert('L')
        mask = Image.open(mask_file[0]).convert('L')

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        bimg_3 = Image.merge('RGB', (img, img, mask))  # binary image into a three-channel image.

        if self.mode == 'train':
            im_aug = transforms.Compose([transforms.RandomApply(
                [transforms.RandomRotation((-90, -90), expand=False, center=None, fill=0)], p=0.5),
                                         transforms.Resize((256, 256))
                                         ])
        else:
            im_aug = transforms.Resize((256, 256))

        img_3 = im_aug(bimg_3)
        r, g, b = img_3.split()
        img = self.preprocess(r, self.scale)
        img = img.repeat(3, 1, 1)
        mask = self.preprocess(b, self.scale)

        return {
            'image': img,
            'mask': mask
        }


