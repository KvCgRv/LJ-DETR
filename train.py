from ultralytics import YOLOv10

# 加载模型
model1 = YOLOv10("yolov10s.yaml")  # 模型结构


if __name__ == '__main__':
    model1.train(data="gc10.yaml", device=0,imgsz=640, batch=32, epochs=200, workers=1)  # 训练模型





