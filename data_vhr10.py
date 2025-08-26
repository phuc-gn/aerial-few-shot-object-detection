import torch
from torch.utils.data import random_split, Subset
from torchgeo.datasets import VHR10

# --- Class splits ---
BASE_CLASSES = list(range(7))       # 0–6
NOVEL_CLASSES = list(range(7, 10))  # 7–9


class VHR10Detection(torch.utils.data.Dataset):
    """Wrapper for TorchGeo VHR10 to match torchvision detection API."""

    def __init__(self, root="data/vhr10", transforms=None, download=True):
        self.dataset = VHR10(root=root, transforms=transforms, download=download)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]

        boxes = sample["bbox_xyxy"].to(torch.float32)        # [N,4]
        labels = sample["label"].to(torch.int64)             # [N]

        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if "mask" in sample:
            target["masks"] = sample["mask"]

        return image, target

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    return tuple(zip(*batch))


def filter_by_classes(dataset, classes):
    """Return subset with only desired classes."""
    indices = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target["labels"].tolist()
        if all(lbl in classes for lbl in labels):
            indices.append(i)
    return Subset(dataset, indices)


def get_vhr10_datasets(root="data/vhr10", fewshot_k=5, seed=42):
    torch.manual_seed(seed)

    full_dataset = VHR10Detection(root=root, transforms=None, download=True)

    # Split 80/20 into train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Base dataset
    base_train = filter_by_classes(train_dataset, BASE_CLASSES)

    # Novel dataset (few-shot: K samples per novel class)
    novel_full = filter_by_classes(train_dataset, NOVEL_CLASSES)
    novel_indices, class_counts = [], {c: 0 for c in NOVEL_CLASSES}
    for i in range(len(novel_full)):
        _, target = novel_full[i]
        for lbl in target["labels"].tolist():
            if lbl in NOVEL_CLASSES and class_counts[lbl] < fewshot_k:
                novel_indices.append(i)
                class_counts[lbl] += 1
                break
    novel_train = Subset(novel_full, novel_indices)

    return base_train, novel_train, test_dataset, collate_fn
