import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    "C:/Users/Sashwat/OneDrive/Desktop/PYTHON/Lesson - Face detection/haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load classifier, check your opencv installation")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("Emotion detection error:", e)

    cv2.imshow("Face & Emotion Detection - Press q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()