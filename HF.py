import cv2
import mediapipe as mp
import time
import numpy as np
import pyttsx3
import threading
from deepface import DeepFace
import matplotlib.pyplot as plt

# Initialize MediaPipe Hand and Face modules.
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Initialize MediaPipe drawing module.
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set the speed of speech

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to classify gestures
def classify_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    gesture = "Unknown"
    if np.linalg.norm(landmarks[mp_hands.HandLandmark.THUMB_TIP] - landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]) < 0.05:
        gesture = "OK"
    elif thumb_tip.y < thumb_ip.y and index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and ring_tip.y > ring_dip.y and pinky_tip.y > pinky_dip.y:
        gesture = "Spiderman"
    elif all(landmarks[tip][1] < landmarks[dip][1] for tip, dip in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
                                                                     (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
                                                                     (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
                                                                     (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)]):
        gesture = "Five"
    elif np.linalg.norm(landmarks[mp_hands.HandLandmark.THUMB_TIP] - landmarks[mp_hands.HandLandmark.PINKY_TIP]) < 0.1:
        gesture = "Rock"
    return gesture

# Start capturing video input from the webcam.
cap = cv2.VideoCapture(0)

# For calculating FPS
prev_time = 0
curr_time = 0

# Initial speech
speak("Welcome to the hand and face detection module")
speak("Please show your hand and face")

hand_detected = False
face_detected = False
prev_gesture = None
prev_emotion = None

# Create a figure for Matplotlib
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while True:
    success, image = cap.read()  # Capture frame-by-frame
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process the image and detect hands and faces.
    hand_results = hands.process(image_rgb)
    face_results = face_detection.process(image_rgb)

    # Convert the image color back so it can be displayed.
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks on the image.
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of the palm (wrist point).
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = image_bgr.shape
            cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
            
            # Draw a circle at the wrist point.
            cv2.circle(image_bgr, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            
            # Display the coordinates on the image.
            cv2.putText(image_bgr, f'Palm Position: ({cx}, {cy})', (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Classify gesture
            gesture = classify_gesture(hand_landmarks)
            cv2.putText(image_bgr, f'Gesture: {gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if not hand_detected:
                speak("Hand detected successfully")
                hand_detected = True

            if gesture != prev_gesture:
                speak(f"Current gesture is {gesture}")
                prev_gesture = gesture

                # Calculate FPS only when gesture is recognized
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                speak(f"Current FPS is {int(fps)}")
    else:
        if hand_detected:
            speak("I don't recognize your hand or hand gesture")
            hand_detected = False

    # Detect faces and emotions
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(image_bgr, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image_bgr.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            # Crop the face from the image
            face_image = image_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            if face_image.size != 0:
                try:
                    emotion_results = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                    if isinstance(emotion_results, list):
                        emotion_results = emotion_results[0]
                    emotion = emotion_results['dominant_emotion']
                    cv2.putText(image_bgr, f'Emotion: {emotion}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    if not face_detected:
                        speak("Face detected successfully")
                        face_detected = True

                    if emotion != prev_emotion:
                        speak(f"Current emotion is {emotion}")
                        prev_emotion = emotion
                except Exception as e:
                    print(f"Error analyzing emotion: {e}")
    else:
        if face_detected:
            speak("I don't recognize your face or emotion")
            face_detected = False

    # Display the image using matplotlib
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))  # Convert to RGB before displaying
    ax.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.01)  # Pause to update the image

    # Close window if the matplotlib figure is closed
    if not plt.fignum_exists(fig.number):
        break

# Release the webcam
cap.release()
plt.close(fig)
