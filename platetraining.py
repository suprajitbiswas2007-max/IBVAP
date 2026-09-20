from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results= model.train(
        data=r"C:\Users\Suprajit\Downloads\plate training\New folder\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="indian_plate_detector"
    )

if __name__ == '__main__':
    main()