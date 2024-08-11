from ultralytics import YOLO

# Initialize a YOLO model (YOLOv5 by default)
model = YOLO('yolov8n.pt')  # You can start with a pre-trained model or use an empty model

# Train the model
results = model.train(data='data.yaml', epochs=20)  # Adjust epochs as needed

# The trained model will be saved in the `runs` directory by default