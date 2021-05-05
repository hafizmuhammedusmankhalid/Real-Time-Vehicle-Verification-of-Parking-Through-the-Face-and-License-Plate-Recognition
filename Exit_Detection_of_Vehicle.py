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