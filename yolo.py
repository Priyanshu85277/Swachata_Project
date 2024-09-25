from ultralytics import YOLO 
model = YOLO('best.pt')  # load a pretrained model (recommended for training)


# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model.predict(source='Test/OIP (5).jpeg',conf=0.25,save=True)
print(results)
