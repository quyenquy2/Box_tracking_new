from ultralytics import YOLO

DATASET_PATH = "configs/tesla_dataset.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=DATASET_PATH,
    imgsz=640,
    epochs=100,
    batch=16,
    device=0,          # GPU
    name="tesla_y11n_gpu"
)

# model.train(
#     data=DATASET_PATH,
#     imgsz=640,
#     epochs=50,
#     batch=4,
#     device="cpu",
#     name="tesla_y11n_cpu"
# )
