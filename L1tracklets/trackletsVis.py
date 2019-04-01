import cv2
import numpy as np
import matplotlib.pyplot as plt  

#def trackletsVis(spatialGroupDetectionCenters,labels):
#    labels = np.ravel(labels)
#    ids = np.unique(labels)
#    fig = plt.figure() 
#    ax1 = fig.add_subplot(111)
#    ax1.set_title('Scatter Plot')  
#    plt.xlabel('X')
#    plt.ylabel('Y') 
#    plt.ylim(0,1080) 
#    plt.xlim(0,1920) 
#    cValue = ['r','y','g','b','r','y','g','b','r'] 
#    for idr in ids:
#        mask = np.where(labels == idr) 
#        centeres_id = spatialGroupDetectionCenters[mask]
#        x = [val[0] for val in centeres_id]
#        y = [val[1] for val in centeres_id]
#        if len(x)<20:
#            continue
#        ax1.scatter(x,y,c=cValue[int(idr)],marker='s')  
#    plt.show()  

def trackletsVis(spatialGroupDetectionCenters,labels):
    labels = np.ravel(labels)
    ids = np.unique(labels)
    image = cv2.imread("L1tracklets/vis.jpg")
    point_color_list =[
        [0, 85, 255], [255, 0, 0],  [255, 170, 0], [255, 255, 0.],[0, 255, 170],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],[0, 255, 85], 
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
        [0, 255, 0],  [0, 255, 255], [0, 170, 255],[255, 85, 0],
    ] 
    for idr in ids:
        mask = np.where(labels == idr) 
        centeres_id = spatialGroupDetectionCenters[mask]
        if(len(centeres_id))<20 :
            continue
        point_list = []
        
        for val in centeres_id:
            cv2.circle(image, (int(val[0]),int(val[1])), 1, point_color_list[int(idr)], 4)
    cv2.imwrite('L1tracklets/vis.jpg',image)
    cv2.imshow('Tracklet',image)
    cv2.waitKey(100)

