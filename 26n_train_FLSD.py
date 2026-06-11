from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")
results = model.train(data="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml", epochs=1, imgsz=640, batch=16)
