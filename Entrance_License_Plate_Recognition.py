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