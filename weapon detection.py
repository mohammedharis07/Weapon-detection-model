import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def load_frame_dataset(dataset_path, sample_limit=None):
    frames, labels = [], []
    class_dirs = {'train': 1, 'valid': 0}

    for class_name, label in class_dirs.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = cv2.imread(file_path)
                if image is None:
                    continue
                image = cv2.resize(image, (128, 128)) / 255.0
                frames.append(image)
                labels.append(label)
            if sample_limit and len(frames) >= sample_limit:
                break

    return np.array(frames), np.array(labels)

# Load YOLOv8 Model with higher confidence threshold
yolo_model = YOLO("yolov8m.pt")  # Using a more accurate model

def detect_objects(frame):
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            confidence = box.conf[0]  
            label = yolo_model.names[int(box.cls[0])]
            
            if label.lower() == "knife" and confidence > 0.4:  # Lower threshold for knife
                color = (0, 0, 255)  # Red for knife
            elif confidence > 0.7:
                color = (0, 255, 0)  # Green for other weapons
            else:
                continue
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Load MediaPipe for Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results.pose_landmarks if results.pose_landmarks else None

def create_feature_extractor():
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_dataset(dataset_path, feature_extractor):
    frames, labels = load_frame_dataset(dataset_path, sample_limit=5000)
    features = np.array([feature_extractor.predict(frame[np.newaxis, ...], verbose=0)[0] for frame in frames])
    
    sequence_length = 30
    sequences, seq_labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i+sequence_length])
        seq_labels.append(labels[i+sequence_length])

    return np.array(sequences), np.array(seq_labels)

def alert_security(frame):
    cv2.putText(frame, "ALERT: Suspicious Activity!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def capture_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return None
    return cap

def main(video_source=0):
    cap = capture_video(video_source)
    if not cap:
        return
    
    feature_extractor = create_feature_extractor()
    lstm_model = create_lstm_model((30, feature_extractor.output_shape[1]))
    sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video source")
            break
        
        frame = detect_objects(frame)  # YOLO Weapon Detection
        detect_pose(frame)  # MediaPipe Pose Detection
        
        frame_resized = cv2.resize(frame, (128, 128)) / 255.0
        features = feature_extractor.predict(frame_resized[np.newaxis, ...], verbose=0)[0]
        sequence.append(features)
        
        if len(sequence) > 30:
            sequence.pop(0)
        
        if len(sequence) == 30:
            sequence_array = np.array(sequence)[np.newaxis, ...]
            prediction = lstm_model.predict(sequence_array, verbose=0)[0][0]
            
            if prediction > 0.7:
                alert_security(frame)
        
        cv2.imshow('Live Surveillance', frame)  # Show camera feed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\Mohammed Haris\OneDrive\Desktop\blahh\weapon-detection.v1i.createml"
    VIDEO_SOURCE = 0  

    print("Training CNN Classifier...")
    feature_extractor = create_feature_extractor()

    print("Preparing LSTM Data...")
    features, labels = preprocess_dataset(DATASET_PATH, feature_extractor)
    if len(features) == 0:
        raise ValueError("No valid sequences found for LSTM training")

    print("Training LSTM Model...")
    lstm_model = create_lstm_model((30, features.shape[2]))
    lstm_model.fit(features, labels, epochs=10, batch_size=8, validation_split=0.2, verbose=1)

    print("Starting Live Surveillance...")
    main(VIDEO_SOURCE)
