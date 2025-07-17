import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import serial
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize serial communication
# Replace 'COM3' with your Arduino's port (check in Arduino IDE > Tools > Port)
ser = serial.Serial('COM7', 9600)
time.sleep(2)  # Wait for Arduino to reset

# Open default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Load dlib face detector
face_detector = dlib.get_frontal_face_detector()

# Load shape predictor
dlib_facelandmark = dlib.shape_predictor(
    r"C:\Users\dell\Documents\shape_predictor_68_face_landmarks.dat"
)

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

# Frame counter and threshold
COUNTER = 0
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
ALERT_TRIGGERED = False  # to control repeated alerts

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1 if n != 47 else 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1 if n != 41 else 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2
        Eye_Rat = round(Eye_Rat, 2)

        if Eye_Rat < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                cv2.putText(frame, "ALERT! WAKE UP!", (50, 450),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

                if not ALERT_TRIGGERED:
                    engine.say("Alert! Wake up!")
                    engine.runAndWait()
                    ser.write(b'1')  # send ON signal to Arduino
                    ALERT_TRIGGERED = True
        else:
            if ALERT_TRIGGERED:
                ser.write(b'0')  # send OFF signal to Arduino
                ALERT_TRIGGERED = False
            COUNTER = 0

        cv2.putText(frame, f"EAR: {Eye_Rat}", (480, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
