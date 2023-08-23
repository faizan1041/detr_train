import torch
from collections import OrderedDict
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
import supervision as sv
from model import *

CHECKPOINT = 'TahaDouaji/detr-doc-table-detection'

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Load the checkpoint file
ckpt_path = 'lightning_logs/version_2/checkpoints/epoch=20-step=819 (copy).ckpt'
checkpoint = torch.load(ckpt_path)

dataset = 'coco/annotations'
TRAIN_DIRECTORY = os.path.join(dataset, "train")
TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY, 
    image_processor=image_processor, 
    train=True)

# utils
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    pixel_images = [transforms.ToPILImage()(img) for img in pixel_values]
    pixel_values = torch.stack(pixel_images)
    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)



# Instantiate your model
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, CHECKPOINT=CHECKPOINT, id2label=id2label, TRAIN_DATALOADER=TRAIN_DATALOADER, VAL_DATALOADER=None)


# Load the checkpoint's state_dict into the model's state_dict
new_state_dict = OrderedDict()
for key, value in checkpoint['state_dict'].items():
    new_key = key.replace('module.', '')  # Remove 'module.' if present (for data parallelism)
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)

# Save the model in .pth format
pth_path = 'epoch=20.pth'
torch.save(model.state_dict(), pth_path)

print("Conversion from ckpt to .pth format completed.")
