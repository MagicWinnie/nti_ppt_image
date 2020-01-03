import numpy as np 
import cv2
from skimage import io
import urllib
# from cv2 import aruco
cv = cv2
# url = str(input())
# url = "https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg"

url = "https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg"
# url = "https://stepik.org/media/attachments/lesson/284187/example_2__-50_-70_.jpg"
# url = "https://stepik.org/media/attachments/lesson/284187/example_3__190_50_.jpg"
# url = "https://stepik.org/media/attachments/lesson/284187/example_4__250_125_.jpg"
# url = "https://stepik.org/media/attachments/lesson/284187/example_5__50_175_.jpg"
def url_to_image(url):
    image = io.imread(url)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



image = url_to_image(url)
# image = cv2.medianBlur(image, 7)
out = image.copy()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# detector_params = cv2.aruco.DetectorParameters_create()
# to_aruco = image.copy()
# to_aruco = cv2.cvtColor(to_aruco, cv2.COLOR_BGR2GRAY)

b_min = np.array((0, 0, 0), np.uint8)
b_max = np.array((255, 255, 150), np.uint8)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = cv2.medianBlur(hsv, 3)
cv2.imshow("hsv", hsv)
to_aruco = cv2.inRange(hsv, b_min, b_max)

# ret,to_aruco = cv.threshold(to_aruco,80,255,cv.THRESH_TRUNC)
to_aruco = cv2.dilate(to_aruco,np.ones((3,3),np.uint8),iterations = 1)
to_aruco = cv2.medianBlur(to_aruco, 3)
to_aruco = cv2.erode(to_aruco,np.ones((3,3),np.uint8),iterations = 1)
cv2.imshow("to_aruco-1", to_aruco)
to_aruco2 = to_aruco.copy()
to_aruco = cv.Canny(to_aruco,2,255)
cv2.imshow("to_aruco", to_aruco)
conts, hs = cv.findContours(to_aruco,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
approx = []
approx_h = []
print("hs", hs[0])
for cnt, h in zip(conts, hs[0]):
    # print("Please")
    aprx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
    
    if len(aprx) == 4 and cv2.contourArea(cnt) > 200 and h[3] == -1 and h[2] != -1:
        # print("ad")
        x,y,w,h = cv.boundingRect(cnt)
        crop = to_aruco2[y+10:y+h-10, x+10:x+w-10]
        print(np.mean(crop))
        if np.mean(crop) > 70:

            approx.append(aprx)
            approx_h.append(h)
# approx_h = [np.array((aprx, ah)) for aprx, ah in [(cv2.approxPolyDP(cnt,0.5*cv2.arcLength(conts[0],True),True), h) for cnt, h in zip(conts, h)] if len(aprx) == 4]
approx_h = np.array([approx_h])
approx = np.array(approx)
print(approx_h)
c_points = []

# to_aruco2 = cv2.erode(to_aruco,np.ones((3,3),np.uint8),iterations = 1)

cv2.drawContours(out,approx,-1,(0,255,0),1)

types = []

for j, cnt in enumerate(approx):
    
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(out,[box],0,(0,150,255),2)

    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)


    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    c_points.append(np.array([cX, cY]))

    for i, p in enumerate(cnt):
        cv2.circle(out, tuple(p[0]), 5, (0, (i+1)*(255//4)//2, (i+1)*(255//4)), -1)
    
    crop = to_aruco2[y:y+h, x:x+w].copy()

    cv2.imshow("crop_" + str(j), crop)
    conts, hs = cv.findContours(crop,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
    aprx = cv2.approxPolyDP(conts[-1],0.02*cv2.arcLength(conts[-1],True),True)
    print(len(aprx))
    # circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 8, 50)
    tp = 0 
    if len(aprx) > 5:
        tp = 1
    cv2.circle(out, (cX, cY), 5, (0, tp*255, 255), -1)
    types.append(tp)

print(types, c_points)
    
# def get_rec_type(aprx, img_in):
#     tp = 0 # 0 - REC    1 - circle
#     # x,y,w,h = cv.boundingRect(cnt)
#     # print(x,y)
#     print(aprx)
#     min_p = min(aprx, key=lambda lmd: (lmd[0][0]**2 + lmd[0][1]**2)**0.5)
#     # min_p = min(aprx, key=lambda lmd: min(lmd[0][0], lmd[0][1]))
#     # aprx_l = aprx[0] - np.array([x,y])
#     aprx_l = np.array([np.array([p[0][0] - min_p[0][0], p[0][1] - min_p[0][1]]) for p in aprx ])
    
#     aprx_l = np.array(sorted(aprx_l, key=lambda lmd: (lmd[0]**2 + lmd[1]**2)**0.5))
#     print("hah", aprx_l)
#     srcTri = np.array(aprx_l).astype(np.float32)
#     dstTri = np.array([[0, 0], [100, 0], [0, 100], [100, 100]]).astype(np.float32)
#     # print("AAAAAAAAAAAAAAAAAAAAAA")
#     # print(srcTri, dstTri)
#     warp_mat = cv.getAffineTransform(srcTri, dstTri)
#     warp_dst = cv.warpAffine(img_in, warp_mat, (100, 100))
#     cv2.imshow("wdp", warp_dst)
#     cv2.waitKey(0)
#     return tp
# get_rec_type(apsprox[0], image)
# approx = approx_h[:, 0]
# h = approx_h[:, 1]
# print(h)
# epsilon = 
# approx = 
# 


# cv2.drawContours(out, conts, -1, (0, 255, 0), -1) #---set the last parameter to -1
# cv2.imshow("to_aruco", to_aruco)
cv2.imshow("out", out)
cv2.waitKey(0)



# corners, ids, rejectedImgPoint = cv2.aruco.detectMarkers(to_aruco, aruco_dict)
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

cv2.imshow("out", out)
# cv2.waitKey(0)


srcTri = np.array(meds).astype(np.float32)
dstTri = np.array([[image.shape[1]//3, image.shape[0]//4*3], [image.shape[1]//3*2, image.shape[0]//4*3], [image.shape[1]//3*2, image.shape[0]//4]]).astype(np.float32)
# print("AAAAAAAAAAAAAAAAAAAAAA")
# print(srcTri, dstTri)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))


hsv = cv2.cvtColor(warp_dst, cv2.COLOR_BGR2HSV)

# cv2.imshow("hsv", hsv)

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


cv2.imshow("warped", warp_dst)
cv2.waitKey(0)

zX = dstTri[0][0]
zY = dstTri[0][1]

imgX = cX - zX
imgY = zY - cY

realX = imgX * (250*3/warp_dst.shape[1])
realY = imgY * (250*2/warp_dst.shape[0])
# print(realX, realY)
realX = int(round(realX, 0))
realY = int(round(realY, 0))

print(realX, realY)
