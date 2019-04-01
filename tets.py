import sys
import cv2
import os
from timeit import time
from external import Deepose

## openpose
Pose = Deepose.DeepPose()

video_capture = cv2.VideoCapture(0)

fps = 0.0
while True:
    ret, frame = video_capture.read()
    if ret != True:
        break;
    t1 = time.time()
    
    keypoints,output_image = Pose.getKeypoints(frame)
    
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= ",fps)
    cv2.imshow(' ',output_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
