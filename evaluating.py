import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from transformers import DetrImageProcessor, DetrForObjectDetection

# Define paths and parameters
model_checkpoint = "model_50_epochs_Aug_8_23:00.pth"  # Path to your trained DETR model checkpoint
data_dir = "coco/annotations"  # Path to your dataset directory

# Register your COCO format dataset (replace "my_dataset" and file paths)
register_coco_instances("my_dataset_train", {}, "coco/annotations/train/_annotations.coco.json", "coco/annotations/train/")
register_coco_instances("my_dataset_val", {}, "coco/annotations/val/_annotations.coco.json", "coco/annotations/val/")

# Load the model
model = DetrForObjectDetection.from_pretrained(model_checkpoint)
processor = DetrImageProcessor.from_pretrained(model_checkpoint)

# Set the model to evaluation mode
model.eval()

# Get the dataset metadata
metadata = MetadataCatalog.get("my_dataset_val")

# Set up the COCO evaluator
coco_evaluator = COCOEvaluator("my_dataset_val", cfg={}, distributed=False, output_dir="./output")

# Load the validation dataset
dataset = DatasetCatalog.get("my_dataset_val")
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Evaluation loop
with torch.no_grad():
    for batch in data_loader:
        inputs = processor(batch)
        outputs = model(**inputs)
        # You can process the model outputs further if needed

# Evaluate and print results
eval_results = inference_on_dataset(model, data_loader, coco_evaluator)
print(eval_results)
