import cv2
import numpy as np
import utilities

# That A4 paper will be our guide in reference to which the length and breadth of the object will be determined

# If we are using image instead of a live webcam
webcam = False
path = "1.jpg"

cap = cv2.VideoCapture(0) # Defining camera object for capturing images. 0 is the id of device

# Setting parameters, like, width, height, brightness, etc.

cap.set(10, 160) # Setting brightness as 160
cap.set(3, 1920) # Setting width as 1920
cap.set(4, 1080) #Setting height as 1080
scaleFactor = 3 #Since our image can be small, we are scaling it by 3
widthOfPaper = 210 * scaleFactor
heightOfPaper = 297 * scaleFactor

while True:
    if webcam:      # If we are capturing image live
        success, img = cap.read()
    else:           # If we are importing image from a file
        img = cv2.imread(path)

    #img, finalContours = utilities.getContours(img, showCanny=True, draw=True)

    imgContours, finalContours = utilities.getContours(img, showCanny=False, minArea=300, filter=4)


    if len(finalContours) != 0:
        biggestContour = finalContours[0][2]
        #print(biggestContour)
        imgWarp = utilities.warpImg(img, biggestContour, widthOfPaper, heightOfPaper)
        # cv2.imshow("A4", imgWarp)

        #Finding objects within imgWarp

        imgContours2, finalContours2 = utilities.getContours(imgWarp, showCanny=False,
                                                             minArea=2000, filter=4, cannyThreshold=[50,50],
                                                             draw=False)
        if len(finalContours) !=0:
            for obj in finalContours2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                #Reordering the points
                nPoints = utilities.reorder(obj[2])
                newWidth = (round(utilities.calcDistance(nPoints[0][0]//scaleFactor,
                                             nPoints[1][0]//scaleFactor)/10),1) #Divided by 10 to convert mm to cm
                # ,1 because we are ronding it to 1 decimal place
                newHeight = (round(utilities.calcDistance(nPoints[0][0]//scaleFactor,
                                             nPoints[2][0]//scaleFactor)/10),1)

                #Adding arrowed lines

                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(newWidth), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(newHeight), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow("A4", imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5) #Resizing the image
    cv2.imshow("Original", img)
    cv2.waitKey(1)

"""
Our first primary focus is to detect the A4 paper.
We already know that the size of an A4 sheet is 297 in height
and 210 in width. Based on that we can find objects on A4 sheet 
and detect their size using pixels.
"""

# We will now use contour method to find the biggest contour
# which will be our paper in this case
