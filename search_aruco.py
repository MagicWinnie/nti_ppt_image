import numpy as np 
import cv2
from skimage import io
import urllib
# from cv2 import aruco
cv = cv2
# url = str(input())
# url = "https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg"

url = "https://stepik.org/media/attachments/lesson/284187/example_1.jpg"

def url_to_image(url):
	image = io.imread(url)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image = url_to_image(url)
out = image.copy()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# detector_params = cv2.aruco.DetectorParameters_create()
# to_aruco = image.copy()
# to_aruco = cv2.cvtColor(to_aruco, cv2.COLOR_BGR2GRAY)

b_min = np.array((0, 0, 0), np.uint8)
b_max = np.array((255, 84, 122), np.uint8)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
to_aruco = 255 - cv2.inRange(hsv, b_min, b_max)

# ret,to_aruco = cv.threshold(to_aruco,80,255,cv.THRESH_TRUNC)
# cv2.imshow("to_aruco", to_aruco)
# cv2.waitKey(0)



corners, ids, rejectedImgPoint = cv2.aruco.detectMarkers(to_aruco, aruco_dict)
# print(corners, ids, rejectedImgPoint)

mrks = sorted(zip(corners, ids), key=lambda x: x[1])
# print(mrks)
ids =  np.array([[101], [102], [103]])
corners = np.array([x[0] for x in mrks])
# print(corners)
cv2.aruco.drawDetectedMarkers(out, corners, ids=ids)
meds = np.mean(corners, axis=2)

# print("BBBBBBBBBBBBBBBBBBbb")
# print(meds)
# cv2.circle(img,(row, col), 5, (0,255,0), -1)

# cv2.imshow("out", out)
# cv2.waitKey(0)


srcTri = np.array(meds).astype(np.float32)
dstTri = np.array([[image.shape[1]//3, image.shape[0]//4*3], [image.shape[1]//3*2, image.shape[0]//4*3], [image.shape[1]//3*2, image.shape[0]//4]]).astype(np.float32)
# print("AAAAAAAAAAAAAAAAAAAAAA")
# print(srcTri, dstTri)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))


hsv = cv2.cvtColor(warp_dst, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv", hsv)

h_min = np.array((0, 75, 75), np.uint8)
h_max = np.array((8, 238, 238), np.uint8)

thresh = cv2.inRange(hsv, h_min, h_max)
thresh = cv2.dilate(thresh,np.ones((5,5),np.uint8),iterations = 1)
thresh = cv2.medianBlur(thresh, 5)

# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)

conts, _ = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
cont = conts[0]

M = cv2.moments(cont)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cv2.circle(warp_dst, (cX, cY), 7, (255, 0, 255), -1)


# cv2.imshow("warped", warp_dst)
# cv2.waitKey(0)

zX = dstTri[0][0]
zY = dstTri[0][1]

imgX = cX - zX
imgY = zY - cY

realX = imgX * (250*3/warp_dst.shape[1])
realY = imgY * (250*2/warp_dst.shape[0])

realX = int(round(realX, 0))
realY = int(round(realY, 0))

print(realX, realY)
