from sklearn.model_selection import train_test_split
import json

def split_dataset(json_path, image_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Load annotations from JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    all_images = data["images"]  # Extract image metadata
    all_annotations = data["annotations"]  # Extract mask annotations

    # Extract image IDs
    image_ids = [img["id"] for img in all_images]

    # Split dataset into train and temp (val + test)
    train_ids, temp_ids = train_test_split(image_ids, train_size=train_ratio, random_state=42)

    # Split temp into validation and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # Adjust ratio for remaining set
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_ratio_adjusted, random_state=42)

    return train_ids, val_ids, test_ids