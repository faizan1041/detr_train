from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
from pdf2image import convert_from_bytes
import supervision as sv
import torch, os, cv2
import numpy as np
from PIL import Image

from model import *

DEVICE = torch.device('cpu')
CHECKPOINT = 'TahaDouaji/detr-doc-table-detection'
CONFIDENCE_THRESHOLD = 0.5
IOU_TRESHOLD = 0.8

os.makedirs("output", exist_ok=True)

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

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

model_weights_paths = "models/model_100.pth"
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, CHECKPOINT=CHECKPOINT, id2label=id2label, TRAIN_DATALOADER=TRAIN_DATALOADER, VAL_DATALOADER=None)
model.load_state_dict(torch.load(model_weights_paths))
model.to(DEVICE)
model.eval()

# inference
image_folder_path = 'input/'
image_files = os.listdir(image_folder_path)
output_folder_path = "output/"

# Inference loop
for image_file in image_files:
    # Load the image using PIL
    image_path = os.path.join(image_folder_path, image_file)
    image_ = cv2.imread(image_path)
    image = cv2.resize(image_,(800,800), interpolation=cv2.INTER_AREA)

    with torch.no_grad():
        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)


        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs, 
            threshold=CONFIDENCE_THRESHOLD, 
            target_sizes=target_sizes
        )[0]
        # print(f'RESULTS: {results}')

    detections = sv.Detections.from_transformers(transformers_results=results)
    # labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=np.array(image), detections=detections, labels=labels)

    # Save the annotated image to the output folder
    output_image_path = os.path.join(output_folder_path, f"aug_{image_file}")
    annotated_image = Image.fromarray(frame)
    annotated_image.save(output_image_path)
