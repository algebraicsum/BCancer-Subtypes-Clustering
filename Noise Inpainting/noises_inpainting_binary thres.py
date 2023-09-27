import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import glob
import time

start_time = time.time()


image_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Data\train\cancer"
output_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Program\Noises Inpainting\result train"

image_files = glob.glob(os.path.join(image_folder, '*.png'))

for path in image_files:
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    (hh, ww) = image.shape[:2]
    
    mean = np.mean(image)
    thresh = cv.threshold(image, mean, 255, 0)[1]
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # get largest contour
    contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)
    big_contour_area = cv.contourArea(big_contour)
    
    mask = np.zeros((hh,ww), dtype=np.uint8)
    for cntr in contours:
        area = cv.contourArea(cntr)
        if area != big_contour_area:
            cv.drawContours(mask, [cntr], 0, 255, cv.FILLED)
    # invert mask
    mask = 255 - mask
    
    # apply mask 
    result = cv.bitwise_and(image, image, mask=mask)
    output_path = os.path.join(output_folder, os.path.basename(path))  
    cv.imwrite(output_path, result)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"elapsed time : {elapsed_time} sec")
# plt.imshow(result,cmap='gray')

# single = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Data\train\cancer\A_1004_1.RIGHT_CC.png"
# image = cv.imread(single, cv.IMREAD_GRAYSCALE)

# (hh, ww) = image.shape[:2]

# mean = np.mean(image)
# thresh = cv.threshold(image, mean, 255, 0)[1]

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
# morph = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel)

# # get largest contour
# contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# big_contour = max(contours, key=cv.contourArea)
# big_contour_area = cv.contourArea(big_contour)

# mask = np.zeros((hh,ww), dtype=np.uint8)
# for cntr in contours:
#     area = cv.contourArea(cntr)
#     if area != big_contour_area:
#         cv.drawContours(mask, [cntr], 0, 255, cv.FILLED)
# # invert mask
# mask = 255 - mask

# # apply mask 
# result = cv.bitwise_and(image, image, mask=mask)
# plt.imshow(result,cmap='gray')
