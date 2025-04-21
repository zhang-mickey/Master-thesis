import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
import torch.nn.functional as F
import numpy as np

import torch.sparse as sparse


class AffinityNet(nn.Module):
    def __init__(self, backbone):
        super(AffinityNet, self).__init__()
        self.backbone = backbone  # This should output feature maps

        self.f_layer2 = torch.nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.f_layer3 = torch.nn.Conv2d(1024, 128, kernel_size=1, bias=False)
        self.f_layer4 = torch.nn.Conv2d(2048, 320, kernel_size=1, bias=False)

        self.f9 = torch.nn.Conv2d(512, 512, 1, bias=False)
        torch.nn.init.kaiming_normal_(self.f_layer2.weight)
        torch.nn.init.kaiming_normal_(self.f_layer3.weight)
        torch.nn.init.kaiming_normal_(self.f_layer4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.from_scratch_layers = [self.f_layer2, self.f_layer3, self.f_layer4, self.f9]
        self.predefined_featuresize = int(512 // 16)

        self.ind_from, self.ind_to = get_indices_of_pairs(5, (self.predefined_featuresize, self.predefined_featuresize))
        self.int_from = torch.from_numpy(self.ind_from)
        self.int_to = torch.from_numpy(self.ind_to)

        return

    def forward(self, x, to_dense=False):
        d = self.backbone(x)  # a list of feature map

        # print("d[2][0] shape:", d[2][0].shape)
        # print("d[2][1] shape:", d[2][1].shape)
        # print("d[2][2] shape:", d[2][2].shape)
        # d[2][0] shape: torch.Size([1, 512, 64, 64])
        # d[2][1] shape: torch.Size([1, 1024, 32, 32])
        # d[2][2] shape: torch.Size([1, 2048, 32, 32])
        f2 = F.elu(self.f_layer2(d[2][0]))
        f2 = F.interpolate(f2, size=(32, 32), mode='bilinear', align_corners=False)

        f3 = F.elu(self.f_layer3(d[2][1]))
        f4 = F.elu(self.f_layer4(d[2][2]))

        x = F.elu(self.f9(torch.cat((f2, f3, f4), dim=1)))  # Shape: (B, 512, H, W)

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from);
            ind_to = torch.from_numpy(ind_to)

        # It flattens all dimensions after the second one.
        # (8, 3, 32, 32)->(8,3,1024)
        x = x.view(x.size(0), x.size(1), -1)

        # print("indices_from shape:", ind_from.shape)
        # print("indices_to shape:", ind_to.shape)
        # indices_from shape: (672,)
        # indices_to shape: (22848,)
        # (B, C, N)
        ind_from = torch.from_numpy(ind_from);
        ind_to = torch.from_numpy(ind_to)
        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        # (batch_size, channels, 1, num_pairs)
        ff = torch.unsqueeze(ff, dim=2)
        ## (B, C, 34, N)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        # torch.abs(ft - ff) → computes the L1 distance between feature vectors
        # torch.mean(..., dim=1) → average across the channel dimension
        # torch.exp(-...) → transforms distance into affinity:
        # Smaller distance ⇒ higher similarity
        # Larger distance ⇒ smaller affinity

        # # (B, 34, N)
        aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                         torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()

            return aff_mat
        else:
            return aff


class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    self.search_dist.append((y, x))

        # creates a safe boundary around the edge of the image.
        self.radius_floor = radius - 1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]

        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        # for each direction (dy, dx)
        for dy, dx in self.search_dist:
            labels_to = label[dy:dy + self.crop_height, self.radius_floor + dx:self.radius_floor + dx + self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)

        concat_labels_to = np.stack(labels_to_list)

        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)),
                                               concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(
            neg_affinity_label)


def get_indices_of_pairs(radius, size):
    # creates a list of (dy, dx) offsets defining which pixels around a central pixel
    # will be considered neighbors.
    search_dist = []
    ## First add horizontal neighbors
    for x in range(1, radius):
        search_dist.append((0, x))
    # Then add circular neighbors (dy, dx) within radius
    # creates a list of (dy, dx) offsets defining which pixels around a central pixel
    # will be considered neighbors.
    search_dist = []
    ## First add horizontal neighbors
    for x in range(1, radius):
        search_dist.append((0, x))

    # Then add circular neighbors (dy, dx) within radius
    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64),
                              (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    # central pixels that will be paired
    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        # for each from pixel, a list of neighboring “to” pixels within the radius
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    # indices_from = np.repeat(indices_from, len(search_dist))
    return indices_from, concat_indices_to


def get_indices_in_radius(height, width, radius):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius - 1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to

