import cv2
import numpy as np

#Read image
image = cv2.imread('line.jpg')

cv2.imshow("Image", image )
cv2.waitKey(0)

#Split channels
b, g, r = cv2.split(image)

#Find green parts
ones = np.ones(g.shape) / 1000000
a = g.astype(np.float64) / (r.astype(np.float64) + b.astype(np.float64) + g.astype(np.float64) + ones)
green = (255 * (a > 0.41)).astype(np.uint8)

#Erode and dilate
kernel = np.ones((13,13), np.uint8)            
opening = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)
morphology = cv2.merge((closing, closing, closing))

gray = cv2.cvtColor(morphology, cv2.COLOR_BGR2GRAY)
(thresh, bnw) = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

cv2.imshow("bnw", bnw * 255)
cv2.waitKey(0)

#Gaussian blur
gauss = cv2.GaussianBlur(image, (3,3), 0)

# Find the edges using canny detector
edges = cv2.Canny(gauss, 50, 100)

cv2.imshow("edge", edges)
cv2.waitKey(0)

#edges in green ground
ground = edges * bnw

cv2.imshow("ground", ground)
cv2.waitKey(0)

#Hough Transform
final_image = image.copy()
lines = cv2.HoughLinesP(ground, 1, np.pi/180, 100, minLineLength=10, maxLineGap=50)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("Result Image", final_image )
cv2.waitKey(0)

#cv2.imwrite('det.jpg', final_image)
