import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q + s, p:p + r]
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(frame, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        cv2.imshow("Output", frame)
        cv2.waitKey(27)
    except cv2.error as e:
        print(f"Error processing frame: {e}")

def main():
    webcam = cv2.VideoCapture(0)
    while True:
        _, frame = webcam.read()
        process_frame(frame)

if __name__ == "__main__":
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    main()
