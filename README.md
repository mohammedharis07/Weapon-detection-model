# Weapon-detection-model

# Weapon Detection and Surveillance System

This project is an AI-powered real-time weapon detection and surveillance system that utilizes YOLOv8, MediaPipe Pose Estimation, and LSTM-based activity recognition. The system can identify weapons, detect human poses, and analyze suspicious activity based on movement patterns.

## Features
- **Weapon Detection**: Uses YOLOv8 to detect weapons like knives and guns in real-time.
- **Pose Estimation**: Implements MediaPipe Pose Estimation to analyze human body positions.
- **Behavior Analysis**: Employs a CNN feature extractor (ResNet50) and an LSTM model to recognize suspicious activities.
- **Alert System**: Triggers an alert overlay when suspicious activity is detected.
- **Real-time Processing**: Captures live video feed and processes frames for detection.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- NumPy
- TensorFlow
- MediaPipe
- UltraLytics YOLOv8

### Install Dependencies
Run the following command to install required packages:
```bash
pip install opencv-python numpy tensorflow mediapipe ultralytics scikit-learn
```

## Usage
### Dataset Preparation
Ensure your dataset is structured as follows:
```
weapon-detection-dataset/
    ├── train/  # Contains images of weapons
    ├── valid/  # Contains images without weapons
```
Modify `DATASET_PATH` in the script accordingly.

### Training the Model
To train the LSTM model:
```bash
python weapon_detection.py
```
This will extract features, create sequences, and train the LSTM model.

### Running Live Surveillance
To start real-time monitoring:
```bash
python weapon_detection.py
```
Press `q` to exit the live surveillance window.

## Code Overview
- **`load_frame_dataset()`**: Loads image frames and assigns labels.
- **`detect_objects()`**: Detects weapons using YOLOv8.
- **`detect_pose()`**: Extracts human body pose using MediaPipe.
- **`create_feature_extractor()`**: Uses ResNet50 to extract deep features.
- **`create_lstm_model()`**: Defines an LSTM network for sequential analysis.
- **`preprocess_dataset()`**: Prepares dataset sequences for LSTM training.
- **`alert_security()`**: Displays an alert overlay.
- **`main()`**: Runs live video surveillance.

## Future Enhancements
- Improve accuracy with more diverse datasets.
- Implement real-time alert notifications via email/SMS.
- Extend support for additional weapons and activities.

## License
This project is open-source and available under the MIT License.

