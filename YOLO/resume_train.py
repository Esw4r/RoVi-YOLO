from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Point directly to the checkpoint file
    model = YOLO(r"runs\detect\train-10\weights\last.pt")

    # 2. Resume training
    model.train(resume=True)

print("Training finished successfully.")