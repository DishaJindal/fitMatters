# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os

MASK_COLOR = (0.0,0.0,1.0)
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
cv2.imshow("original",image);

 #fgmask = fgbg.apply(image)

# cv2.imshow('fgmask',image)
# cv2.imshow('frame',fgmask)

size = image.shape[:2] 
h = size[0]-1
w = size[1]-1
print(h, w)
colorTopLeft = image[0,0]
colorTopRight = image[0,w]
diff = np.subtract(colorTopRight.astype(np.int16), colorTopLeft.astype(np.int16))
weightedTopColor = np.divide(diff.astype(np.float64), float(w))

# image = cv2.copyMakeBorder(image, 10, 0, 0, 0, cv2.BORDER_CONSTANT, value= colorTop.tolist())

for y in range(0, 10):
	prev = colorTopLeft.astype(np.float64)
	for x in range(0, w):
		prev = np.add(prev.astype(np.float64), weightedTopColor.astype(np.float64))
		image[y, x] = prev

colorBottomLeft = image[h,0]
colorBottomRight = image[h,w]
diff2 = np.subtract(colorBottomRight.astype(np.int16), colorBottomLeft.astype(np.int16))
weightedBottomColor = np.divide(diff2.astype(np.float64), float(w))

for y in range(h-10, h+1):
	prev = colorBottomLeft.astype(np.float64)
	for x in range(0, w):
		prev = np.add(prev.astype(np.float64), weightedBottomColor.astype(np.float64))
		image[y, x] = prev		

# colorBottom = image[size[0]-1, size[1]-1]
# image = cv2.copyMakeBorder(image, 0, 10, 0, 0, cv2.BORDER_CONSTANT, value= colorBottom.tolist())

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(image, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("afterEdgeDetection",edged);

edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("afterDilation",edged);

edged = cv2.erode(edged, None, iterations=1)
 
cv2.imshow("afterErosion", edged);
cv2.waitKey(0)




# Face recognition logic

cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

image_face = image.copy()

maxFaceHeight = 0
maxFaceWidth = 0
FaceDepth = 0

for (x, y, w, h) in faces:
	if h >= maxFaceHeight:
		maxFaceHeight = h
		maxFaceWidth = w
		FaceDepth = y + h
	print("face width in pixels: ", w)
	print("face height in pixels: ", h)
	cv2.rectangle(image_face, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image_face)
cv2.waitKey(0)





# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually

# for c in cnts:
# 	# if the contour is not sufficiently large, ignore it
# 	print cv2.contourArea(c)

selected_height = 1
selected_width = 1

count = 1
ppm = 1
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	# print cv2.contourArea(c)
	if cv2.contourArea(c) < 400:
		continue

 
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

 
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
 
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
 
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
 
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
 
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	print(dA)
	print(dB)

	if dA > selected_height:
		selected_height = dA
		selected_width = dB
		tl1 = tl
		tr1 = tr
		br1 = br
		bl1 = bl
		box1 = box
 
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]
		ppm = pixelsPerMetric
# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	print(count)
	print("height: " , dimA , "shoulder width: ", dimB)
	count = count+1
 
	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
 
	# show the output image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)

final_img = image.copy()
tl1[1] = max(tl1[1], 1.2*FaceDepth)
tr1[1] = max(tr1[1], 1.2*FaceDepth)
print ("ppm",ppm)
ch = ppm*5
tl1[0] = tl1[0] + ch
tr1[0] = tr1[0] - ch
bl1[0] = bl1[0] + ch
br1[0] = br1[0] - ch

cv2.drawContours(final_img, [np.array([tl1, tr1, br1, bl1]).astype("int")], -1, (0, 255, 0), 2)
(tltrX1, tltrY1) = midpoint(tl1, tr1)
(blbrX1, blbrY1) = midpoint(bl1, br1)
(tlblX1, tlblY1) = midpoint(tl1, bl1)
(trbrX1, trbrY1) = midpoint(tr1, br1)
cv2.circle(final_img, (int(tltrX1), int(tltrY1)), 5, (255, 0, 0), -1)
cv2.circle(final_img, (int(blbrX1), int(blbrY1)), 5, (255, 0, 0), -1)
cv2.circle(final_img, (int(tlblX1), int(tlblY1)), 5, (255, 0, 0), -1)
cv2.circle(final_img, (int(trbrX1), int(trbrY1)), 5, (255, 0, 0), -1)
cv2.line(final_img, (int(tltrX1), int(tltrY1)), (int(blbrX1), int(blbrY1)), (255, 0, 255), 2)
cv2.line(final_img, (int(tlblX1), int(tlblY1)), (int(trbrX1), int(trbrY1)), (255, 0, 255), 2)
dA1 = dist.euclidean((tltrX1, tltrY1), (blbrX1, blbrY1))
dB1 = dist.euclidean((tlblX1, tlblY1), (trbrX1, trbrY1))
print(dA1)
print(dB1)
dimA1 = dA1 / ppm 
dimB1 = dB1 / ppm 

cv2.putText(final_img, "{:.1f}cm".format(dimA1), (int(tltrX1 - 15), int(tltrY1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
cv2.putText(final_img, "{:.1f}cm".format(dimB1), (int(trbrX1 + 10), int(trbrY1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

print("face height: ", float(maxFaceHeight)/ppm, "  face width : ", float(maxFaceWidth)/ppm)
print("actual height: " , dimA1 , " actual shoulder width: ", dimB1)

cv2.imwrite('processed.png', final_img)
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + "/processed.png"
file = open("hw.txt","w")
file.write(str(dimA1))
file.write("\n")
file.write(str(dimB1))
file.write("\n")
file.write(dir_path)
file.close()

 # if pixelsPerMetric is None:
 # 		pixelsPerMetric = dB / args["width"]

 # resizedHeight = selected_height/ pixelsPerMetric
 # resizedWidth = selected_width / pixelsPerMetric

# print("body height: ", resizedHeight, " width: ", resizedWidth)

# resizedHeight = (selected_height - maxFaceHeight)/ pixelsPerMetric

# print("body - face height: ", resizedHeight, " width : ", resizedWidth)
