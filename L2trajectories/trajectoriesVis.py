import cv2
import numpy as np
import matplotlib.pyplot as plt  

def trajectoriesVis(Trajectories):
    image = cv2.imread("L2trajectories/vis.jpg")
    point_color_list = [[0, 85, 255], [255, 0, 0], [255, 170, 0], [255, 255, 0.],[0, 255, 170],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],[0, 255, 85], 
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
        [0, 255, 0],  [0, 255, 255], [0, 170, 255],[255, 85, 0]] 
    
    for ind in range(len(Trajectories)):
        trajectory = Trajectories[ind]
        tracklets = trajectory['tracklets']
        for intra in range(len(tracklets)):
            centers = tracklets[intra]['data']
            for bbx in centers:
                cv2.circle(image, (int(bbx[2]+0.5*bbx[4]),int(bbx[3]+0.5*bbx[5])), 1, point_color_list[ind], 4)
                
    cv2.imwrite('L2trajectories/vis.jpg',image)
    cv2.imshow('Tracklet',image)
    cv2.waitKey(100)
    
