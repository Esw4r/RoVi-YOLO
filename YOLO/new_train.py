from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"C:\Users\eshwar\Documents\CLG\SEM-7\NN-DL\Gopher-AI\rdd2022\data.yaml",
        epochs=100,
        patience=10,         # Stops early if mAP doesn't improve for 10 epochs
        imgsz=512,           # 36% fewer FLOPs per batch than 640
        batch=16,             # Keeps FP32 tensors safely inside 4GB VRAM
        workers=3,
        cache=True,          # Dataset fits in RAM now (~1.5GB); zero disk lag
        amp=False,           # Required for GTX 1650 stability
        degrees=0.0,
        flipud=0.0,
        fliplr=0.5,
        close_mosaic=10,
        optimizer="AdamW",
        lr0=0.0015,
        save=True
    )