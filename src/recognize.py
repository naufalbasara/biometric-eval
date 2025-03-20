import logging, argparse, sys, os, pendulum
import cv2, numpy as np
import tensorflow as tf

from datetime import datetime
from numpy.linalg import norm
from utils.db_conn import PostgrePy

localtz = pendulum.timezone("Asia/Jakarta")

def detect_faces(img):
    """Detect faces using haarcascade classifier from OpenCV, returning coordinates for bounding box"""
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        return x,y,w,h


def get_vector(image_dir, classifier, detect_faces):
    img = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    try:
        x,y,w,h = detect_faces(img)
        cropped = img[y-125:y+h+75, x-125:x+w+75]
    except:
        logging.warning(f"Face not found in {image_dir}")
        cropped = img

    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'log/registered_{datetime.today()}.png', resized)
    preprocessed = np.expand_dims(resized, 0)
    user_vector = classifier.predict(preprocessed, verbose=0)

    return user_vector

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A)*norm(B))

def register_user(pgcon, user:dict, vector):
    username = user.get('name', None)
    if username == None:
        return False

    try:
        pgcon.cur.execute("INSERT INTO users (name, embedding, registered_at) VALUES (%s, %s, %s)", (username, vector.tolist(), datetime.now(tz=localtz)))
        return True
    except Exception as error:
        print(f"Failed to register user due to: {error}")
    
    return False

def find_closest(pgcon, vector, threshold=0.8):
    try:
        # pgcon.cur.execute("SELECT name, embedding <=> %s as distance FROM users2", (f'{vector.tolist()}',))
        pgcon.cur.execute(f"SELECT name FROM users2 where embedding <=> %s <= {1-threshold}", (f'{vector.tolist()}',))
        data = {'header': [i[0] for i in pgcon.cur.description], 'data': pgcon.cur.fetchall()}
    except Exception as error:
        print(f"Failed to retrieve users with input face:")
        print(error)

    return data

def predict_from_cam_stream():
    # recognition phase from camera stream
    # Start camera to stream and detect face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    print("Start streaming frame..")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x-150,y-150), (x+w+100, y+h+100), (255,0,0),2)

        if cv2.waitKey(1) == 32:
            captureImage = frame[y-125:y+h+75, x-125:x+w+75]
            preprocessed = cv2.resize(captureImage, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'log/captured_{datetime.today()}.png', preprocessed)
            preprocessed = np.expand_dims(preprocessed, 0)
            res = model.predict(preprocessed, verbose=0).flatten()
            break

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', "--mode", required=True, help="'register', 'recognize', or 'compare' mode")
    parser.add_argument('-src', "--source", help='Source of user face image (file path) or camera stream (default)')
    parser.add_argument('-src2', "--source2", help='Source of user face image (file path) or camera stream (default) (2)')
    parser.add_argument('-u', "--username", help='User name for registration')
    
    args = parser.parse_args()
    mode = getattr(args, 'mode')
    src = getattr(args, 'source')
    src2 = getattr(args, 'source2')
    username = getattr(args, 'username')

    if mode == 'register' and username == None:
        parser.error("'register' mode requires -u username.")

    if mode == 'compare' and (src == None or src2 == None):
        parser.error("'compare' mode requires two user face images to be input (src & src2)")

    try:
        model = tf.keras.models.load_model('pre-trained/vggface2.h5')
        print("Model loaded.")
    except Exception as error:
        raise error
    
    try:
        pgcon = PostgrePy()
    except Exception as error:
        raise error
    
    if mode == 'register':
        # registration phase from user's face image
        if src != None:
            img_dir = src
            user_vector = get_vector(img_dir, model)
            user_vector = user_vector.flatten()
        else:
            user_vector = predict_from_cam_stream()

        registered = register_user(
                pgcon,
                {'name': username}, 
                user_vector
            )
        print(f"User registered status: {registered}")
    
    elif mode == 'recognize':
        # recognition phase from user's face image
        if src != None:
            img_dir = src
            user_vector = get_vector(img_dir, model)
            user_vector = user_vector.flatten()
        else:
            user_vector = predict_from_cam_stream()

        resp = find_closest(pgcon=pgcon, vector=user_vector)
        print(resp)

    elif mode == 'compare':
        # recognition phase from camera stream
        # # Start camera to stream and detect face
        user1 = get_vector(src, classifier=model)
        user2 = get_vector(src2, classifier=model)
        sim = cosine_similarity(user1.flatten(), user2.flatten())
        print('User similarity ==> ', sim)
    

