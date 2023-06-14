import numpy as np
import cv2
import time


class GetEdgeLocs:
    def __init__(self):
        print("Class to get image boundary locations")

    def getLocs(self, mask):
        h, w = mask.shape
        img = np.ones((h, w))
        img[mask] = 0
        img = np.uint8(img * 255)

        #make edges smooth
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        img_cont = np.zeros((h, w, 3), np.uint8)
        all_contours = []
        i = 0
        print("computing boundaries")
        while img.any() > 0:
            contours, hier = cv2.findContours(
                img, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE)  #cv.CHAIN_APPROX_SIMPLE
            #cv2.drawContours(img_cont, contours, 3, (0, 255, 25), 10)
            visited = np.concatenate(contours).squeeze(axis=1)
            img[(visited[:, 1], visited[:, 0])] = 0
            all_contours.append(visited)
        print("computing boundaries done")

        return all_contours
