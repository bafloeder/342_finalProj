# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
# 	help="path to the first image")
# ap.add_argument("-s", "--second", required=True,
# 	help="path to the second image")
# args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

first = "images/IMG_5735.JPG"
second = "images/IMG_5737.JPG"
third = "images/IMG_5738.JPG"

imageA = cv2.imread(first)
imageB = cv2.imread(second)
imageC = cv2.imread(third)

# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB, imageC], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
# cv2.imwrite("matches.jpg", vis)
cv2.imshow("Result", result)
cv2.imwrite("result.jpg", result)
cv2.waitKey(0)