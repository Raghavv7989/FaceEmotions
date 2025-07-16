import cv2
import mediapipe as mp
from deepface import DeepFace
import threading
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variable for shared emotion result
latest_emotion = "Detecting..."
last_stable_emotion = "Detecting..."
last_update_time = time.time()

# Function to analyze emotion in a background thread
def analyze_emotion(frame_copy):
    global latest_emotion, last_stable_emotion, last_update_time
    try:
        result = DeepFace.analyze(frame_copy, actions=['emotion'], enforce_detection=False)
        detected_emotion = result[0]['dominant_emotion']

        # Smooth out emotion changes
        current_time = time.time()
        if detected_emotion == latest_emotion or current_time - last_update_time > 1.5:
            last_stable_emotion = detected_emotion
            last_update_time = current_time

        latest_emotion = detected_emotion
    except Exception:
        latest_emotion = "Detecting..."


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Start a background thread for emotion analysis (less frequent)
    if int(time.time() * 10) % 15 == 0:
        frame_copy = frame.copy()
        threading.Thread(target=analyze_emotion, args=(frame_copy,), daemon=True).start()

    # Face mesh detection
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Display the stable emotion
    cv2.putText(frame, f"Emotion: {last_stable_emotion}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Smooth 3D Face Mesh + Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
