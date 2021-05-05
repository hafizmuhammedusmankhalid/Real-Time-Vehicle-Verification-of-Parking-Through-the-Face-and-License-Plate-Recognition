# Real-Time Vehicle Verification of Parking Through the Face and License Plate Recognition

A world that has a lot of car users, parking management and its security is quite complicated issues. A secure parking system that is free of larceny, is considered a luxury now. The system works on computer vision techniques and is designed as the automatic vehicle identifier concerning the vehicle driver and license plate identification. A Facial recognition algorithm to verify the driver and an Automatic number plate recognition algorithm to distinguish the license plate is applied through machine learning and deep learning techniques. A facial recognition library recognizes the face of the driver that is entering the parking area from the images. The system is processed in Python three and is divided into three steps. The first step will capture the picture of the car while entering the parking site and will record it as a comparative data source that will employ the system to identify the driver and number plate while exiting the parking. Secondly, an extra layer is added to security in case of a mismatch in the face of the driver. The printer generates an invoice having a QR code on it. Thirdly, the system will notify the operators of the parking area to take measures in case of a mismatch. The unique idea of this identification  system is to achieve better results by using multiple cameras and two-way authentication. This system reckons reducing the number of car thefts committed in parking places. It also forbids the entrance of those vehicles and drivers who are unauthorized or not allowed by the respective firm.

## Entrance_Conversion_of_Video_to_Frames.py

```

import cv2
vidcap = cv2.VideoCapture(r"Entrance_Video\Entrance_Video.mp4")
success,image = vidcap.read()
width = 1000
height = 700
dim = (width, height)
count = 0
while success:
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("Entrance_Video_Frames/Entrance_Video_Frame%d.jpg" % count, resized)    # save frame as JPG file      
    success,image = vidcap.read()
    count += 1

```

This code is used to convert the video into frames with a width of 1000 and a height of 700. The following is one of the frame which is converted from video using above code.

![](\Entrance_Video_Frames\Entrance_Video_Frame0.jpg)

## Entrance_Detection_of_Vehicle.py

```

import os
import sys
import cv2
import numpy as np

# Loading of Model

net = cv2.dnn.readNet(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Weights.weights", r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Model.cfg")

classes = []

with open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
os.chdir(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Video_Frames")

for images in os.listdir():
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    cap = cv2.imread(images)
    frame_id = 0
    while True:
        frame = cap
        frame_id += 1

        height, width, channels = frame.shape

        # Detecting Car
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:

                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
        for i in range(len(boxes)):
            region_of_interest = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            region_of_interest = region_of_interest[y:y+h, x:x+w]
            cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Vehicle_Detected\Entrance_Vehicle_Detected.jpg", region_of_interest)
            cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Vehicle_Detected\Vehicle_Detected.jpg", frame)
            os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Vehicle_Detected\Vehicle_Detected.jpg")
            sys.exit()

```

This code is used to detect the vehicle. The following image is an example of vehicle detection using the above code.

![](\Entrance_Vehicle_Detected\Vehicle_Detected.jpg)

## Entrance_License_Plate_Recognition.py

```

import os
import cv2
import numpy as np
import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = cv2.imread("Entrance_License_Plate\Entrance_License_Plate.jpg")               # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates
    
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates
    
    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        list = []
        num = len(listOfPossiblePlates)
        for i in range(0,num):
            licPlate = listOfPossiblePlates[i]
            list.append(licPlate.strChars)
            i+=1
        recognition_of_license_plate = list[1]+list[0]    
        file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "a")
        file. truncate(0)
        file.write(recognition_of_license_plate)
        file.close()
        
        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        
        image = cv2.imread(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Vehicle_Detected\Entrance_Vehicle_Detected.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, recognition_of_license_plate,(150, 150), font, 1,(0, 255, 0), 3)
        cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate_Recognition.jpg", image)
        os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate_Recognition.jpg")
        
if __name__ == "__main__":
    main()

```

This code is used to detect the license plate of the vehicle and apply optical character recognition to it and fetch characters from it. The following image is an example of vehicle license plate recognition using the above code.

![](\Entrance_License_Plate\Entrance_License_Plate_Recognition.jpg)

## Encoding_of_QR_Code_and_Generating_Invoice_With_QR_Code.py

```

import qrcode
import os
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()


qr = qrcode.QRCode(
    version=1,
    box_size=5,
    border=1
)

data = recognition_of_license_plate
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')
img.save(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Encoding_of_QR_Code\\' + 'QR_Code' + '.jpg')

today = datetime.today()
current_day = today.strftime("%B %d, %Y")
current_time = today.strftime("%H:%M:%S")

canvas = canvas.Canvas(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Invoice_With_QR_Code\Invoice_With_QR_Code.pdf")
canvas.setPageSize((300, 420))

qr_code = ImageReader(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Encoding_of_QR_Code\QR_Code.jpg')
canvas.setLineWidth(.3)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(20, 390,'Real time vehicle verification through face and ')
canvas.drawString(85, 375, 'license plate recognition')
canvas.line(1, 365, 299, 365)
canvas.drawString(115, 350,'Parking Invoice')
canvas.setFont('Helvetica', 12)
canvas.drawString(7, 310,'Reciept Number: 1')
canvas.drawString(7, 285,'License Plate: '+str(data))
canvas.drawString(160, 310, 'Date: ' + str(current_day))
canvas.drawString(160, 285, 'Entrance Time: ' + str(current_time))
canvas.line(1, 275, 299, 275)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(130, 260,'QR-Code')
canvas.drawImage(qr_code, 100, 140)
canvas.line(1, 130, 299, 130)
canvas.drawString(125, 115,'Instructions')
canvas.setFont('Helvetica', 12)
canvas.drawString(7, 100,'1. Place this slip in front of the scanner at')
canvas.drawString(20, 85,'the time of exit.')
canvas.drawString(7, 70,'2. This invoice is valid for one time only.')
canvas.drawString(7, 55,'3. If the invoice is misplaced. Please immediately')
canvas.drawString(20, 40, 'contact the security.')
canvas.line(1, 30, 299, 30)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(110, 15,'Thanks for visitng')
canvas.save()
os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Invoice_With_QR_Code\Invoice_With_QR_Code.pdf")

```

This code is used to generate an invoice with an encoded QR code.

## Entrance_Face_Detection.py

```

import os 
import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Video_Frames\\')

count = 0

for images in os.listdir():
    image = cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    
    if len(result) > 0:
        count = count + 1
        if count < 25:
            continue
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        roi = cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 3)
        cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Face_Detected\User1.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Face_Detected\User1.jpg")
        break 
    else:
        continue

```

This code is used to detect the vehicle driver's face. The following image is an example of vehicle driver's face using the above code.

![](\Entrance_Face_Detected\User1.jpg)

## Entrance_Play_Sound.py

```

import os
from playsound import playsound

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Play_Sound\\')

if not recognition_of_license_plate:
    playsound('Entrance_Not_Allowed_Play_Sound.mp3')
else:
    playsound('Entrance_Allowed_Play_Sound.mp3')

```

This code is used to play sound.

## Entrance_Record_Vehicle.py

```

import os
import pandas as pd
from datetime import datetime

today = datetime.today()
current_day = today.strftime("%B %d, %Y")
current_time = today.strftime("%H:%M:%S")

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()

city = pd.DataFrame([[current_day, current_time, recognition_of_license_plate]], columns=['Vehicle Entrace Date', 'Vehicle Entrance Time', 'Vehicle License Plate'])
city.to_csv(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')

os.system(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')

```

This code is used to generate a CSV of the entrance record of the vehicle with its license plate.

## Exit_Conversion_of_Video_to_Frames.py

```

import cv2
vidcap = cv2.VideoCapture(r"Exit_Video\Exit_Video.mp4")
success,image = vidcap.read()
width = 1000
height = 700
dim = (width, height)
count = 0
while success:
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("Exit_Video_Frames/Exit_Video_Frame%d.jpg" % count, resized)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1


```

This code is used to convert the video into frames with a width of 1000 and a height of 700. The following is one of the frame which is converted from video using above code.

![](\Exit_Video_Frames\Exit_Video_Frame0.jpg)

## Exit_Detection_of_Vehicle.py

```

import os
import sys
import cv2
import numpy as np

# Loading of Model

net = cv2.dnn.readNet(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Weights.weights", r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Model.cfg")

classes = []

with open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Car_Detection_Model\Classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
os.chdir(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Video_Frames")

for images in os.listdir():
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    cap = cv2.imread(images)
    frame_id = 0
    while True:
        frame = cap
        frame_id += 1

        height, width, channels = frame.shape

        # Detecting Car
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:

                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    
        for i in range(len(boxes)):
            region_of_interest = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            region_of_interest = region_of_interest[y:y+h, x:x+w]
            cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Vehicle_Detected\Exit_Vehicle_Detected.jpg", region_of_interest)
            cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Vehicle_Detected\Vehicle_Detected.jpg", frame)
            os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Vehicle_Detected\Vehicle_Detected.jpg")
            sys.exit()         

```

This code is used to detect the vehicle. The following image is an example of vehicle detection using the above code.

![](\Exit_Vehicle_Detected\Vehicle_Detected.jpg)

## Exit_License_Plate_Recognition.py

```

import os
import cv2
import numpy as np
import DetectChars
import DetectPlates
import PossiblePlate

font = cv2.FONT_HERSHEY_SIMPLEX

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = cv2.imread("Exit_License_Plate/Exit_License_Plate.jpg")               # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates
    
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates
    
    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        list = []
        num = len(listOfPossiblePlates)
        for i in range(0,num):
            licPlate = listOfPossiblePlates[i]
            list.append(licPlate.strChars)
            i+=1
        exit_recognition_of_license_plate = list[1]+list[0] 
        
        file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
        entrance_recognition_of_license_plate = file.read()
        
        if exit_recognition_of_license_plate == entrance_recognition_of_license_plate:
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Checking_Authorization_License_Plate.txt", "a")
            file. truncate(0)
            file.write("1")
            file.close()
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Exit_License_Plate.txt", "a")
            file. truncate(0)
            file.write(exit_recognition_of_license_plate)
            file.close()
         
        else:
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Checking_Authorization_License_Plate.txt", "a")
            file. truncate(0)
            file.write("0")
            file.close() 
            
        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        
        image = cv2.imread(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Vehicle_Detected\Exit_Vehicle_Detected.jpg")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, exit_recognition_of_license_plate,(130, 150), font, 1,(0, 255, 0), 3)
        cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Exit_License_Plate_Recognition.jpg", image)
        os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Exit_License_Plate_Recognition.jpg")

        
if __name__ == "__main__":
    main()

```

This code is used to detect the license plate of the vehicle and apply optical character recognition to it and fetch characters from it. The following image is an example of vehicle license plate recognition using the above code.

![](\Exit_License_Plate\Exit_License_Plate_Recognition.jpg)


## Exit_Face_Detection.py

```

import os 
import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Video_Frames\\')

count = 0

for images in os.listdir():
    image = cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    
    if len(result) > 0:
        count = count + 1
        if count < 25:
            continue
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        roi = cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 3)
        cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Test_Face.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Test_Face.jpg")
        break
        
    else:
        continue

```

This code is used to detect the vehicle driver's face. The following image is an example of vehicle driver's face using the above code.

![](\Exit_Face_Detected\Test_Face.jpg)

## Exit_Face_Recognition.py

```

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

```

This code is used for facial recognition. This code also checking the authorization of the vehicle driver's face.

## Exit_Decoding_of_QR_Code.py

```

import sys
import cv2
import time
import numpy as np
from pyzbar.pyzbar import decode

#cap = cv2.VideoCapture("http://192.168.18.20:8080/video")

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)
check = 0 
    
with open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt") as f:
    myDataList = f.read().splitlines()

while True:
 
    success, img = cap.read()
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)
 
        if myData in myDataList:
            myOutput = 'Authorized'
            myColor = (0,255,0)
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "a")
            file. truncate(0)
            file.write("1")
            check = 1
            file.close()
        
        else:
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)
            myOutput = 'Authorized'
            myColor = (0,255,0)
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "a")
            file.truncate(0)
            file.write("0")
            check = 2
            file.close()

            
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,myColor,5)
        pts2 = barcode.rect
        cv2.putText(img,myOutput,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,myColor,2)
 
    cv2.imshow('Decoding of Quick Response Code', img)
    cv2.waitKey(1)
    
    if check == 1 or check == 2:
        time.sleep(10)
        cv2.destroyAllWindows()
        sys.exit()

```

This code is used to check the authorization of the QR code.

## Exit_Play_Sound.py

```

import os
from playsound import playsound

file0 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Checking_Authorization_License_Plate.txt", "r")
Checking_Authorization_License_Plate = file0.read()

print(Checking_Authorization_License_Plate)

file1 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Checking_Authorization_Face.txt", "r")
Checking_Authorization_Face = file1.read()

print(Checking_Authorization_Face)

file2 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "r")
Checking_Authorization_of_QR_Code = file2.read()           

print(Checking_Authorization_of_QR_Code)

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Play_Sound\\')

if (Checking_Authorization_License_Plate == "1" and Checking_Authorization_Face == "1" and Checking_Authorization_of_QR_Code == "1") or (Checking_Authorization_License_Plate == "1" and Checking_Authorization_of_QR_Code == "1"):
    playsound('Exit_Allowed_Play_Sound.mp3')
else:
    playsound('Exit_Not_Allowed_Play_Sound.mp3')


```

This code is used to play sound.

## Exit_Record_Vehicle.py

```

import os
import pandas as pd
from datetime import datetime

today = datetime.today()
current_day = today.strftime("%B %d, %Y")
current_time = today.strftime("%H:%M:%S")

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Exit_License_Plate.txt", "r")
recognition_of_license_plate = file.read()

city = pd.DataFrame([[current_day, current_time, recognition_of_license_plate]], columns=['Vehicle Entrace Date', 'Vehicle Entrance Time', 'Vehicle License Plate'])
city.to_csv(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')

os.system(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')

```

This code is used to generate a CSV of the entrance record of the vehicle with its license plate.