import cv2
import numpy as np
import matplotlib.pyplot as plt  
import random

def trajectoriesVis_det(detections):

    image = cv2.imread("data/vis.jpg")
    
    detections = np.array(detections)
    personIDs = np.unique(detections[:,1])
    
    for i in range(len(personIDs)):
        personID = personIDs[i]
        
        color = [random.randint(0,255) for i in range(3)]
        
        person_det = np.where(detections[:,1]==personID)
        relevantDetections = detections[person_det]
        for det in relevantDetections:
            if(det[2]==0 and det[3]==0):
                continue
            cv2.circle(image, (int(det[2]+0.5*det[4]),int(det[3]+0.5*det[5])), 1, color, 4)
                
    cv2.imwrite('L2trajectories/vis_det.jpg',image)
    cv2.imshow('Tracklet_det',image)
    cv2.waitKey(100)
    
