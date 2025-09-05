from ultralytics import YOLO
import torch

if __name__ == "__main__":
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO('yolov8m.pt')

    model.train(
        data=r"C:\Users\Admin\Desktop\output_yolo_dataset\remapped\data.yaml",
        project=r'C:\Users\Admin\Desktop\trash_yolo_training',
        name='yolov8m_trash',
        device = device,
        epochs=100,
        imgsz=640,
        batch=8,
        lr0=0.01,         # initial learning rate
        lrf=0.1,          # final OneCycleLR learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1
    )

    metrics = model.val()
    print(metrics)

    results = model.predict(r"C:\Users\Admin\Desktop\output_yolo_dataset\remapped\images\test\000037_9.JPG",save=True)
    results[0].show()