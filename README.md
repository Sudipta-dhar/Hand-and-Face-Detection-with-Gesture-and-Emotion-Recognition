# Hand and Face Detection with Gesture and Emotion Recognition

This Python program detects and analyzes hand gestures, face detection, and emotions in real-time using webcam input. The program utilizes MediaPipe for hand and face detection, DeepFace for emotion analysis, and pyttsx3 for text-to-speech feedback. The processed results are displayed on a live video feed using OpenCV and Matplotlib.

## Requirements

Before running the program, install the following Python libraries:

- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- Pyttsx3 (`pyttsx3`)
- DeepFace (`deepface`)
- Matplotlib (`matplotlib`)
- Numpy (`numpy`)

You can install them via `pip`:

```bash
pip install opencv-python mediapipe pyttsx3 deepface matplotlib numpy
```

## Program Features
- Hand Gesture Recognition: Detects and classifies gestures such as "OK", "Spiderman", "Five", and "Rock" using hand landmarks detected by MediaPipe.
- Face Detection and Emotion Recognition: Detects faces and analyzes emotions (e.g., happy, sad, angry) using DeepFace.
- Text-to-Speech Feedback: Provides real-time audio feedback about detected hand gestures and face emotions using pyttsx3.
- Interactive Display: Displays results on a live video feed with OpenCV and Matplotlib, showing landmarks and gestures on the webcam input.

## How It Works
- Webcam Input: Captures real-time video frames from the webcam.
- Hand Detection: Detects hand landmarks and classifies hand gestures based on finger positions.
- Face Detection: Detects faces and analyzes emotions using DeepFace on the detected face region.
- Gesture and Emotion Recognition: The program classifies hand gestures and identifies emotions from the face.
- Text-to-Speech: The program speaks the recognized gestures and emotions using pyttsx3.
- Display Results: The video feed shows landmarks and classified gestures/emotions on the webcam input.
- Instructions for Running the Program
## Launch the Program: 
Run the Python script, and the webcam will open. The program will begin detecting hand gestures and faces.

## Instructions for Running the Program
Launch the Program: Run the Python script, and the webcam will open. The program will begin detecting hand gestures and faces.

```bash
python hand_face_detection.py
```
- Provide Input: Position your hand and face in front of the webcam. The program will detect the hand and face, classify gestures, and recognize emotions.

- Output: The results (gestures and emotions) will be displayed on the live video feed, and the program will announce them via speech.

- Close the Program: Close the video window to stop the program, or use Ctrl+C in the terminal.

## Program Output
Real-time Video Feed: A webcam video feed is displayed, with detected hand landmarks and facial bounding boxes overlaid.

- Speech Output: The program announces the following:
"Hand detected successfully" or "Face detected successfully"
- Recognized gesture (e.g., "OK", "Spiderman", "Rock")
- Detected emotion (e.g., "Happy", "Sad")
- FPS Calculation: The program calculates and speaks the FPS (frames per second) based on gesture recognition.

Example Speech Output
"Welcome to the hand and face detection module"
"Please show your hand and face"
"Hand detected successfully"
"Current gesture is OK"
"Current emotion is Happy"
"Current FPS is 15"
Key Functions
speak(text)
Converts text to speech using pyttsx3 and provides feedback to the user.

### classify_gesture(hand_landmarks)
Analyzes the hand landmarks and classifies the gesture based on the relative positions of the hand landmarks.

Main Loop
The main loop captures video frames, processes them for hand and face detection, displays the results, and provides feedback on gestures and emotions.

Troubleshooting
Camera Not Working: Ensure your camera is properly connected and not being used by another application.
Low FPS: Reduce the resolution of the webcam feed or close other resource-intensive applications.
Emotion Analysis Not Working: Test DeepFace separately to confirm it's working properly.

## References

1. **OpenCV**  
   OpenCV is an open-source computer vision and machine learning software library that provides tools for real-time image processing.  
   - OpenCV: https://opencv.org/

2. **MediaPipe**  
   MediaPipe is a framework developed by Google for building cross-platform pipelines for processing multimodal data, such as video, audio, and other sensor data. It was used in this project for hand and face detection.  
   - MediaPipe: https://google.github.io/mediapipe/

3. **pyttsx3**  
   pyttsx3 is a text-to-speech conversion library in Python that provides speech synthesis. It was used in this project to give real-time feedback for gestures and emotions.  
   - pyttsx3: https://pypi.org/project/pyttsx3/

4. **DeepFace**  
   DeepFace is a Python library for deep learning-based facial recognition and emotion analysis. It was used in this project to analyze emotions from detected faces.  
   - DeepFace: https://github.com/serengil/deepface

5. **Matplotlib**  
   Matplotlib is a plotting library for Python which provides an object-oriented API for embedding plots into applications. It was used in this project to display video frames and overlay results.  
   - Matplotlib: https://matplotlib.org/

6. **NumPy**  
   NumPy is a fundamental package for scientific computing in Python, used for handling arrays and matrices. It was utilized to process hand landmarks and gestures.  
   - NumPy: https://numpy.org/

7. **Python Documentation**  
   - Official Python Documentation: https://docs.python.org/

### License
This project is licensed under the MIT License.




