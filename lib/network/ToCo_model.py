import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.network.ToCo_decoder_conv_head import *
from . import ToCo_encoder as encoder

"""
Borrow from https://github.com/facebookresearch/dino
"""


class CTCHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ToCo_network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.proj_head = CTCHead(in_dim=self.encoder.embed_dim, out_dim=1024)
        self.proj_head_t = CTCHead(in_dim=self.encoder.embed_dim, out_dim=1024, )

        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data.copy_(param.data)  # initialize teacher with student
            param_t.requires_grad = False  # do not update by gradient

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [
                                                                                                       self.encoder.embed_dims[
                                                                                                           -1]] * 4

        self.pooling = F.adaptive_max_pool2d

        # self.decoder = LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes,)
        self.decoder = LargeFOV(in_planes=self.in_channels[-1], out_planes=2, )

        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=1, kernel_size=1, bias=False,)

        # self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=1, kernel_size=1, bias=False,)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=2, kernel_size=1, bias=False, )

        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=2, kernel_size=1, bias=False, )

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self, n_iter=None):
        ## no scheduler here
        momentum = self.init_momentum
        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.proj_head.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward_proj(self, crops, n_iter=None):

        global_view = crops[:2]
        local_view = crops[2:]

        local_inputs = torch.cat(local_view, dim=0)

        self._EMA_update_encoder_teacher(n_iter)

        global_output_t = self.encoder.forward_features(torch.cat(global_view, dim=0))[0].detach()
        output_t = self.proj_head_t(global_output_t)

        global_output_s = self.encoder.forward_features(torch.cat(global_view, dim=0))[0]
        local_output_s = self.encoder.forward_features(local_inputs)[0]
        output_s = torch.cat((global_output_s, local_output_s), dim=0)
        output_s = self.proj_head(output_s)

        return output_t, output_s

    def forward(self, x, cam_only=False, crops=None, n_iter=None):

        cls_token, _x, x_aux = self.encoder.forward_features(x)

        if crops is not None:
            output_t, output_s = self.forward_proj(crops, n_iter)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)

        seg = self.decoder(_x4)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()

            return cam_aux, cam

        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)

        # cls_x4 = cls_x4.view(-1, 1)
        # cls_aux = cls_aux.view(-1, 1)
        cls_x4 = cls_x4.view(-1, 2)
        cls_aux = cls_aux.view(-1, 2)
        if crops is None:
            return cls_x4, seg, _x4, cls_aux
        else:
            return cls_x4, seg, _x4, cls_aux, output_t, output_s


def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


# Region of Interest mask
def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    # Expand cls_label to shape [B, C, H, W] so it matches the shape of cam.
    if cls_label.ndim == 1:
        cls_label = F.one_hot(cls_label, num_classes=c).float()

    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2

    return roi_mask


def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):
    crops = []

    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)
    margin = crop_size // 2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h - margin), margin:(w - margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_mask = roi_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            if temp_mask.sum() / (crop_size * crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2 + 2] = 0

    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1, )
    crops = [c[:, 0] for c in _crops]

    return crops, flags


def cam_to_label(cam, cls_label,
                 # img_box=None,
                 bkg_thre=None,
                 high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    ## Mask CAMs with class labels
    valid_cam = cls_label_rep * cam
    # Get pseudo-labels from CAMs
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index
    pseudo_label = _pseudo_label

    return valid_cam, pseudo_label


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    ## Ignore diagonal
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None,
                            images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False,
                            # img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)

    cls_labels = cls_labels.unsqueeze(1) if cls_labels.dim() == 1 else cls_labels

    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx in range(b):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...],
                                        cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...],
                                        cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx] = _refined_label_h[0]
        refined_label_l[idx] = _refined_label_l[0]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label