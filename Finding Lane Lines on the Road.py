
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys


def GetImageWithCannyEdgeDetection(imageBase, low_threshold, high_threshold):

    # Getting gray image.
    grayImage = cv2.cvtColor(imageBase, cv2.COLOR_RGB2GRAY)

    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size     =   5 # Must be an odd number (3, 5, 7...)
    blur_grayImage  =   cv2.GaussianBlur(grayImage, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and run it
    imageResult    = cv2.Canny(blur_grayImage, low_threshold, high_threshold)

    return  imageResult;


def GetImageMasked(imageBase):

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask              = np.zeros_like(imageBase)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape  = imageBase.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_edges = cv2.bitwise_and(imageBase, mask)

    return  masked_edges;



def GetImageWithHoughLines(imageBase, distanceResolution, theta, threshold, minLineLenght, maxlineGap):
    
    edges               =   GetImageWithCannyEdgeDetection(imageBase, 50, 150)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask              = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape           = imageBase.shape
    vertices          = np.array([[(0,imshape[0]),(460, 315), (490, 315), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges      = cv2.bitwise_and(edges, mask)

    # Make a blank the same size as our image to draw on
    line_image        = np.copy(imageBase)*0    # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines             = cv2.HoughLinesP(masked_edges, distanceResolution, theta, threshold, np.array([]), minLineLenght, maxlineGap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10) # BGR

    # Create a "color" binary image to combine with line image
    color_edges       = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges       = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

    return  lines_edges;


def GenerateOutPutFile(inputFileName, pathFilePath):

    inputFullFileName   =   pathFilePath + inputFileName + '.mp4'
    outputFullFileName  =   pathFilePath + inputFileName + '_result.mp4'

    vcVideoInput  =   cv2.VideoCapture(inputFullFileName)
 
    if vcVideoInput.isOpened():
        rval , imageBase = vcVideoInput.read()
    else:
        rval = False

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width  = int(vcVideoInput.get(3))
    frame_height = int(vcVideoInput.get(4))

    # Define the fps to be equal to 10. Also frame size is passed.
    fourcc          =   cv2.VideoWriter_fourcc(*'MP4V')
    vcVideoOutput   =   cv2.VideoWriter(outputFullFileName, fourcc, 15.0, (frame_width, frame_height))
    counter         =   1

    while(rval):

        imageFrame  =   GetImageWithHoughLines(imageBase, 1, np.pi/180, 15, 40, 20)

        vcVideoOutput.write(imageFrame)

        counter = counter + 1

        rval, imageBase = vcVideoInput.read()

    vcVideoOutput.release()
    vcVideoInput.release()

#
# Begin
# 
GenerateOutPutFile('solidWhiteRight', './')
GenerateOutPutFile('solidYellowLeft', './')
GenerateOutPutFile('challenge',       './')

sys.exit(0)
