#Instead of directly mapping input → label, construct pairs/triplets of similar and dissimilar samples.
# 	•	image_id: Refers to the image to which this annotation belongs.
# 	•	category_id: Refers to the class (e.g., high-opacity-smoke, low-opacity-smoke).
# 	•	area: Represents the area of the object in the image.
# 	•	iscrowd: Indicates if the object is part of a crowd (this might not be directly needed for your contrastive learning task).

import random
import json
from torch.utils.data import Dataset
from PIL import Image
import warnings


class SmokeContrastiveDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transform=None):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        self.images_folder = images_folder
        self.transform = transform
        self.image_annotations = {image['id']: [] for image in self.data['images']}

        for annotation in self.data['annotations']:
            self.image_annotations[annotation['image_id']].append(annotation['category_id'])

        # Map category_id to supercategory
        self.category_to_supercategory = {category['id']: category['supercategory'] for category in
                                          self.data['categories']}
        # Convert to binary labels (smoke vs non-smoke)
        self.image_labels = {}

        for image_id, annotations in self.image_annotations.items():
            # For each image, determine if it belongs to "smoke" or "non-smoke"
            label = 0  # default is non-smoke
            for category_id in annotations:
                supercategory = self.category_to_supercategory[category_id]
                if supercategory == "smoke":
                    label = 1  # If any category is of "smoke", label as smoke
                    break
            self.image_labels[image_id] = label

        # Store image IDs and their corresponding categories
        self.image_ids = list(self.image_annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image_id for the current sample
        image_id = self.image_ids[index]
        label = self.image_labels[image_id]

        # Load the image
        image_path = f"{self.images_folder}/{self.data['images'][image_id]['file_name']}"
        image = Image.open(image_path)
        anchor_label = label
        # Get a positive sample (same class)
        positive_indices = [id for id in self.image_ids if self.image_labels[id] == anchor_label and id != image_id]
        print("num of positve",len(positive_indices))

        if len(positive_indices) == 0:
            warnings.warn(f"No positive sample for image_id {image_id}. Using self-pairing.")
            positive_image_id = image_id  # Use anchor itself
        else:
            positive_image_id = random.choice(positive_indices)


        positive_image_path = f"{self.images_folder}/{self.data['images'][positive_image_id]['file_name']}"
        positive_image = Image.open(positive_image_path)

        # Get a negative sample (different class)
        negative_indices=[id for id in self.image_ids if self.image_labels[id] != anchor_label]

        if len(negative_indices) == 0:
            warnings.warn(f"No negative sample for image_id {image_id}. Using self-pairing.")
            negative_image_id = image_id  # Use anchor itself
        else:
            negative_image_id = random.choice(negative_indices)


        negative_image_path = f"{self.images_folder}/{self.data['images'][negative_image_id]['file_name']}"
        negative_image = Image.open(negative_image_path)

        if self.transform:
            image = self.transform(image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return image, positive_image, negative_image,anchor_label