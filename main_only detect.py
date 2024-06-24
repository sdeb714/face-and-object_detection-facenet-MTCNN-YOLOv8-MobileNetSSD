import cv2
from ultralytics import YOLO
import mtcnn
from architecture import *
from train import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import numpy as np
import os
import pickle
import csv
from datetime import datetime
from collections import defaultdict, Counter
import pygame
import threading

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
unknown_images_folder = "unknown_images_folder"
last_attendance_time = defaultdict(lambda: None)
last_unknown_save_time = None
unknown_save_interval = 10  # seconds
pygame.mixer.init()

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def markAttendance(name):
    global last_attendance_time
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%d-%B-%Y')

    if (name not in last_attendance_time) or ((now - last_attendance_time[name]).total_seconds() >= 60):
        with open('known_info.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_str, date_str])

        last_attendance_time[name] = now

def save_unknown_info(frame, face, encoding, date_str, time_str):
    global last_unknown_save_time
    now = datetime.now()
    if last_unknown_save_time is None or (now - last_unknown_save_time).total_seconds() >= unknown_save_interval:
        img_name = f'{unknown_images_folder}/unknown_{date_str}_{time_str}.jpg'
        cv2.imwrite(img_name, frame)

        with open('unknown_info.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['unknown', time_str, date_str, img_name])

        last_unknown_save_time = now

def play_sound():
    sound_file_path = "Sound/aleart.mp3"
    pygame.mixer.music.load(sound_file_path)
    pygame.mixer.music.play()

def show_saved_images(folder):
    while True:
        unknown_images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        for img_name in unknown_images:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

    cv2.destroyAllWindows()

def run_show_saved_images(folder):
    thread = threading.Thread(target=show_saved_images, args=(folder,))
    thread.start()

def detect(img, detector, encoder, encoding_dict, save_path="unknown_images_folder", display_saved_image=True):
    known_face_count = 0
    unknown_face_count = 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    saved_image = None

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            unknown_face_count += 1
            now = datetime.now()
            date_str = now.strftime('%d-%B-%Y')
            time_str = now.strftime('%H-%M-%S')
            save_unknown_info(img, face, encode, date_str, time_str)
            play_sound()

            saved_image_path = f'{unknown_images_folder}/unknown_{date_str}_{time_str}.jpg'
            # saved_image = cv2.imread(saved_image_path)

        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            markAttendance(name)
            known_face_count += 1

    cv2.putText(img, f'Known: {known_face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Unknown: {unknown_face_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show saved unknown images

    # if saved_image is not None:
    #     cv2.imshow('Saved Image', saved_image)
    #     cv2.waitKey(1000)

    return img

def detect_people_gathering(img, net, threshold=1):
    height, width, _ = img.shape
    target_size = (300, 300)
    resized_img = cv2.resize(img, target_size)
    blob = cv2.dnn.blobFromImage(resized_img, 0.007843, target_size, 127.5)
    net.setInput(blob)
    detections = net.forward()
    people_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:
                people_count += 1

    if people_count >= threshold:
        cv2.putText(img, f'People Gathering: {people_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img, people_count

if __name__ == "__main__":
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    mobilenet_ssd_model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
    prediction_model = YOLO("mobilepen.pt")
    class_names = ['mobile', 'pen']
    cap = cv2.VideoCapture(0)
    cap_saved_image = cv2.VideoCapture(1)
    run_show_saved_images(unknown_images_folder)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict)
        frame, people_count = detect_people_gathering(frame, mobilenet_ssd_model, threshold=1)
        results = prediction_model(frame)

        object_counts = Counter()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0]
                class_name = class_names[cls]
                object_counts[class_name] += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        y_offset = 150
        for class_name, count in object_counts.items():
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

        cv2.imshow('Live Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap_saved_image.release()
    cv2.destroyAllWindows()
