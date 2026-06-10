from ultralytics import YOLO

model = YOLO("yolo26n.pt")

result = model("ultralytics/assets/bus.jpg")
