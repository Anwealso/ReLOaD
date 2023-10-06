# !pip install -U ultralytics

import torch
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['/home/alex/Pictures/brisbane1.jpg',
        '/home/alex/Pictures/brisbane2.jpg',
        '/home/alex/Pictures/brisbane3.jpg',
        '/home/alex/Pictures/brisbane4.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()