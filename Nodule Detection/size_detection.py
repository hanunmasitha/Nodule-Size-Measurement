# import the necessary packages
import warnings

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import pandas as pd

warnings.filterwarnings(action='ignore')

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

PREDICT_DIR = r'E:\Koding\Python\Thesis\Nodule Size Detection\Predict Segmentation\Raw Image'
ORIGINAL_DIR = r'E:\Koding\Python\Thesis\Nodule Size Detection\Predict Segmentation\PNG Image'
COUNT_DIR = r'E:\Koding\Python\Thesis\Nodule Size Detection\Predict Segmentation\Predict Size'
META_DIR = r'E:\Koding\Python\Thesis\Nodule Size Detection\Predict Segmentation\Meta'
meta_size = pd.DataFrame(index=[],columns=['patient_id','image name','size'])

width = 1.55

for file in os.listdir(PREDICT_DIR):
    file_code = os.path.splitext(file)[0]

    #read file
    image = PREDICT_DIR + "\\" + file
    image = np.load(image)

    #normalize file from 0-1 to 0-255
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    cv2.imwrite(norm_image + '\\' + file_code + '.png', orig)

    #detect edge
    edged = cv2.Canny(norm_image, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if(cnts):
        cv2.imwrite(ORIGINAL_DIR + '\\' + file_code + '.png', norm_image)
        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / width

            # compute the size of the object
            dimA = (dA / pixelsPerMetric) * 2.54
            dimB = (dB / pixelsPerMetric) * 2.54
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}cm".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}cm".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

            # save image
            cv2.imwrite(COUNT_DIR + '\\' + file_code + '.png', orig)

            if(dimA > dimB):
                size = dimA
            else:
                size = dimB

            patienID = file_code.split('_')[0]

            meta_list = [patienID, file_code, size]
            print(meta_list)

            tmp = pd.Series(meta_list, index=['patient_id','image name','size'])
            meta_size = meta_size.append(tmp, ignore_index=True)

meta_size.to_csv(META_DIR+'\meta_info.csv',index=False)