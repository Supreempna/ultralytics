from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.yaml")

results = model.train(data="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml", epochs=12, imgsz=640, batch=16)
