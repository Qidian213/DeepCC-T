
import numpy as np  
import cv2  
#create a black use numpy,size is:512*512
img = np.zeros((1080,1920,3), np.uint8)   
img.fill(255)
cv2.imshow('image', img)  
cv2.imwrite('vis.jpg',img)

