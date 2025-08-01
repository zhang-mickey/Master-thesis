from pycocotools.coco import COCO
import numpy as np
import os
import cv2

# json_path = 'annotations/instances_train2014.json'
# img_dir = 'train2014'
# save_mask_dir = 'masks/train2014'
json_path = 'annotations/instances_val2014.json'
img_dir = 'train2014'
save_mask_dir = 'masks/val2014'
os.makedirs(save_mask_dir, exist_ok=True)

coco = COCO(json_path)
catid2label = {cat['id']: i for i, cat in enumerate(coco.loadCats(coco.getCatIds()))}

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    height, width = img_info['height'], img_info['width']

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    label_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        if ann.get('iscrowd', 0):
            continue
        mask = coco.annToMask(ann)
        class_id = catid2label[ann['category_id']]
        label_mask[mask == 1] = class_id

    save_name = os.path.splitext(file_name)[0] + '.png'  # '000000581929.png'
    save_path = os.path.join(save_mask_dir, save_name)
    cv2.imwrite(save_path, label_mask)


# from pycocotools.coco import COCO

# json_path = "annotations/instances_train2014.json"
# output_txt = "train.txt"

# coco = COCO(json_path)
# img_ids = coco.getImgIds()

# with open(output_txt, "w") as f:
#     for img_id in img_ids:
#         f.write(f"{str(img_id).zfill(12)}\n")  # COCO 图像名格式为 12 位数字

# json_path = "annotations/instances_val2014.json"
# output_txt = "val.txt"

# coco = COCO(json_path)
# img_ids = coco.getImgIds()

# with open(output_txt, "w") as f:
#     for img_id in img_ids:
#         f.write(f"{str(img_id).zfill(12)}\n")