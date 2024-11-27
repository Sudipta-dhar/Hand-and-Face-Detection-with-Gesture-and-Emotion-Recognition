# Hand-and-Face-Detection-with-Gesture-and-Emotion-Recognition
This Python program uses various libraries to detect and analyze hand gestures, face detection, and emotions in real-time using a webcam feed. The program employs MediaPipe for hand and face detection, DeepFace for emotion analysis, and pyttsx3 for text-to-speech feedback. It displays the results on a live video feed using OpenCV and Matplotlib.

Requirements
Before running the program, ensure you have the following libraries installed:

OpenCV (opencv-python)
MediaPipe (mediapipe)
Pyttsx3 (pyttsx3)
DeepFace (deepface)
Matplotlib (matplotlib)
Numpy (numpy)
You can install these libraries via pip:

bash
Copy code
pip install opencv-python mediapipe pyttsx3 deepface matplotlib numpy
Program Features
Hand Gesture Recognition: Detects and classifies hand gestures like "OK", "Spiderman", "Five", and "Rock" using hand landmarks detected by MediaPipe.

Face Detection and Emotion Recognition: Identifies faces and analyzes emotions (e.g., happy, sad, angry) using DeepFace, based on the face region detected.

Text-to-Speech Feedback: The program gives real-time feedback via speech using pyttsx3, notifying users about detected hand gestures and face emotions.

Interactive Display: The results are displayed on a live video feed using OpenCV and Matplotlib, with landmarks and gestures shown on the webcam input.

How It Works
Webcam Input: The program uses your webcam to capture real-time video frames.

Hand Detection: It detects hand landmarks, draws them on the frame, and classifies hand gestures based on the positions of the fingers.

Face Detection: It detects faces within the frame and extracts the region of the face for emotion analysis using DeepFace.

Gesture and Emotion Recognition: The program classifies gestures such as "OK", "Spiderman", "Rock", etc., based on hand landmarks. It also identifies emotions such as happiness, sadness, or anger based on the face.

Text-to-Speech: The program speaks out the current gesture and emotion using pyttsx3.

Display Results: The processed video feed with the drawn landmarks, gesture, and emotion is displayed on a live feed.

Instructions for Running the Program
Launch the Program: Simply run the Python script. The webcam will open, and the program will begin detecting hands and faces, recognizing gestures, and analyzing emotions.

bash
Copy code
python hand_face_detection.py
Provide Input: Position your hand and face in front of the webcam. The program will detect your hand and face and classify gestures and emotions.

Output: The results (gestures and emotions) are displayed on the live video feed, and the program will announce them using text-to-speech.

Close the Program: To close the program, simply close the video window, or interrupt the process using Ctrl+C in the terminal.

Program Output
Real-time Video Feed: A webcam video feed is displayed, with detected hand landmarks and facial bounding boxes overlaid.

Speech Output: The program will announce when a hand gesture or face is detected and will provide the recognized gesture or emotion. The speech includes:

"Hand detected successfully" or "Face detected successfully".
The recognized gesture (e.g., "OK", "Spiderman", "Rock").
The detected emotion (e.g., "Happy", "Sad").
FPS Calculation: The program will also calculate and speak the FPS (frames per second) based on gesture recognition.

Example Speech Output
"Welcome to the hand and face detection module"
"Please show your hand and face"
"Hand detected successfully"
"Current gesture is OK"
"Current emotion is Happy"
"Current FPS is 15"
Key Functions
speak(text)
This function is responsible for converting text to speech using the pyttsx3 library. It is called whenever the program needs to speak feedback to the user.

classify_gesture(hand_landmarks)
This function analyzes the hand landmarks detected by MediaPipe and classifies the gesture based on the relative positions of the hand landmarks.

Main Loop
The main loop captures the video frame, processes it for hand and face detection, and displays the results using OpenCV. It also performs gesture and emotion classification and provides feedback via speech.

Troubleshooting
Camera Not Working: Ensure that your camera is connected properly and not being used by another application.
Low FPS: If the FPS is low, try reducing the resolution of the webcam feed or closing other resource-heavy applications.
Emotion Analysis Not Working: Ensure that DeepFace is correctly installed and working by testing it independently.

License
This program is provided under the MIT License.







