from ultralytics import YOLO

model = YOLO("runs/detect/tesla_y11n/weights/best.pt")
model.export(format="onnx")
