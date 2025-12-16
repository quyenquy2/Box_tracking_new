import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from ultralytics import YOLO
print("Ultralytics OK")

from roboflow import Roboflow
print("Roboflow OK")
