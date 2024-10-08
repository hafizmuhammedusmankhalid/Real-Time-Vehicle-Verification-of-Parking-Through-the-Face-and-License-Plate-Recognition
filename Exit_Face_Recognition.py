import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

string2 = ['Unknown']

def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./Entrance_Face_Detected"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("Entrance_Face_Detected/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded


def unknown_image_encoded(img):
    
    face = fr.load_image_file("Entrance_Face_Detected/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:
        cv2.imshow('Recognition of Face', img)
        cv2.waitKey(0)
        return face_names 
        break
            
string = classify_face(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Test_Face.jpg")

if string == string2:
    file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Checking_Authorization_Face.txt", "a")
    file. truncate(0)
    file.write("0")
    file.close()
else:
    file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Checking_Authorization_Face.txt", "a")
    file. truncate(0)
    file.write("1")
    file.close()
             
cv2.destroyAllWindows()