import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from transforms import *
import torchvision
from PIL import Image
from torchvision import transforms as T
import random
import torchvision.transforms.functional as TF

class_list = ["bg", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
              'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'tvmonitor']


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str, usecols=0)
    return img_name_list


# def load_cls_label_list(name_list_dir):

#     return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

def load_cls_label_list_from_txt(txt_path, num_classes=21):
    label_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_name = parts[0]
            labels = list(map(int, parts[1:])) if len(parts) > 1 else []
            onehot = np.zeros(num_classes, dtype=np.uint8)
            onehot[labels] = 1
            label_dict[img_name] = onehot
    return label_dict


class VOC12Dataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        if split == 'train':
            split_file = 'train_aug_num.txt'  # use augmented list
        else:
            split_file = split + '.txt'

        self.name_list_dir = os.path.join(name_list_dir, split_file)
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = str(self.name_list[idx])
        img_name = os.path.join(self.img_dir, _img_name + '.jpg')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:, :, 0]

        return _img_name, image, label


class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        # self.color_jittor = transforms.PhotoMetricDistortion()

        self.gaussian_blur = GaussianBlur
        self.solarization = Solarization(p=0.2)

        # self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.label_list = load_cls_label_list_from_txt(os.path.join(name_list_dir, 'train_cls.txt'))
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=1.0),
            # self.normalize,
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        local_image = None
        if self.aug:

            if self.rescale_range:
                image = random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = random_fliplr(image)
            if self.crop_size:
                image, img_box = random_crop(image, crop_size=self.crop_size, mean_rgb=[0, 0, 0],
                                             ignore_index=self.ignore_index)

            local_image = self.local_view(Image.fromarray(image))
            image = self.global_view1(Image.fromarray(image))

        image = self.normalize(image)

        return image, local_image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)

        pil_image = Image.fromarray(image)

        image, local_image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.aug:

            crops = []
            crops.append(image)
            crops.append(self.global_view2(pil_image))
            crops.append(local_image)
            # for _ in range(8):
            #     crops.append(self.local_view(pil_image))

            return img_name, image, cls_label, img_box, crops
        else:
            return img_name, image, cls_label


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = PhotoMetricDistortion()
        if stage == "train":
            label_file = "train_cls.txt"
        elif stage == "val":
            label_file = "val_cls.txt"
        elif stage == "test":
            label_file = "test_cls.txt"
            # self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.label_list = load_cls_label_list_from_txt(os.path.join(name_list_dir, label_file))

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label, _ = random_crop(image, label, crop_size=self.crop_size,
                                              mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''

        image = TF.resize(Image.fromarray(image), size=(512, 512))
        label = TF.resize(Image.fromarray(label), size=(512, 512), interpolation=Image.NEAREST)

        image = np.array(image)
        label = np.array(label)

        image = normalize_img(image)
        image = np.transpose(image, (2, 0, 1))  # to CHW

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label