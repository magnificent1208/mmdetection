import cv2


rect1 = ((50,50), (100,100), 0)  # x,y  w,h 
rect2 = ((50,40), (50,200), 60)
 
r1 = cv2.rotatedRectangleIntersection(rect1, rect2)  # 区分正负角度，逆时针为负，顺时针为正
import pdb; pdb.set_trace() 
order_pts = cv2.convexHull(r1[1], returnPoints=True)
int_area = cv2.contourArea(order_pts)
