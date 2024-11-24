import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

# Path to the folder containing known images
KNOWN_IMAGES_FOLDER = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces and their names
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for file_name in os.listdir(KNOWN_IMAGES_FOLDER):
        image_path = os.path.join(KNOWN_IMAGES_FOLDER, file_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file_name)[0])
    return known_face_encodings, known_face_names

# Mark attendance in the CSV file
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])
    
    with open(ATTENDANCE_FILE, mode='r') as file:
        attendance_data = file.readlines()
        if any(name in line for line in attendance_data):
            return  # Avoid duplicate marking

    with open(ATTENDANCE_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date_str, time_str])

# Main function
def main():
    print("Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces()

    print("Starting video capture...")
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

                # Draw a rectangle around the face
                top, right, bottom, left = face_location
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the video feed
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
