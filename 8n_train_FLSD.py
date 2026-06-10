from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

result = model.train(data="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml", epochs=12, imgsz=640,batch=16)

