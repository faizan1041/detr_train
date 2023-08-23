from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
import supervision
import transformers
import pytorch_lightning
from pytorch_lightning import Trainer
import torch
import os

# from model import CocoDetection
from model import *

import torch
# !nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print(
    "; supervision:", supervision.__version__,
    "; transformers:", transformers.__version__,
    "; pytorch_lightning:", pytorch_lightning.__version__
)

# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'TahaDouaji/detr-doc-table-detection'

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

dataset = 'coco/annotations'
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "val")
# TEST_DIRECTORY = os.path.join(dataset, "annotations/test")

TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    image_processor=image_processor,
    train=True)
VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    image_processor=image_processor,
    train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
print(len(id2label))



from PIL import Image
import torchvision.transforms as transforms

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    # print(f'LABELS: {labels}')
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
    
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=8, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=8)


model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, CHECKPOINT=CHECKPOINT, id2label=id2label, TRAIN_DATALOADER=TRAIN_DATALOADER, VAL_DATALOADER=VAL_DATALOADER)
batch = next(iter(TRAIN_DATALOADER))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

outputs.logits.shape

MAX_EPOCHS = 100

trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, 
                  accumulate_grad_batches=8, log_every_n_steps=5)
# trainer.fit(model)
trainer.fit(model)
model.to(DEVICE)
model.eval()
model_path = f"model_100_epochs.pth"
torch.save(model.state_dict(),model_path)
