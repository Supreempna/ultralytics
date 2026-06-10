from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

result = model.train(data="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml", epochs=300, imgsz=640)
