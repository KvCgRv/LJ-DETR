from ultralytics import RTDETR
 
if __name__ == '__main__':
    model = RTDETR('/mnt/d/subuntu/steel_thing/ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml')
    model.train(data='coal_rock.yaml',
                imgsz=640,
                epochs=200,
                batch=8,
                workers=0,
                device="0",
                optimizer='SGD',  # 可以使用的优化器：SGD和AdamW
                project="yolov10"
                )


#nohup python RT_DETR_run.py > train_logs/coal_wtq2detrs_b32.log 2>&1 &