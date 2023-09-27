import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import glob
import time

start_time = time.time()


image_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Data\train\cancer"
output_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Program\Noises Inpainting\otsu train"

image_files = glob.glob(os.path.join(image_folder, '*.png'))

for path in image_files:
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    (hh, ww) = image.shape[:2]
    
    mean = np.mean(image)
    thresh = cv.threshold(image, 0, 255, cv.THRESH_OTSU)[1]
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)
    
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    # morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel)
    
    # get largest contour
    contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)
    # big_contour_area = cv.contourArea(big_contour)
    
    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh,ww), dtype=np.uint8)
    cv.drawContours(mask, [big_contour], 0, 255, cv.FILLED)
    
    # dilate mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31,31))
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
    
    result = cv.bitwise_and(image, image, mask=mask)
    output_path = os.path.join(output_folder, os.path.basename(path))  
    cv.imwrite(output_path, result)

end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"elapsed time : {elapsed_time} sec")
# plt.imshow(result,cmap='gray')

# single = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Program\Noises Inpainting\otsu test\A_1933_1.LEFT_MLO.png"
# image = cv.imread(single, cv.IMREAD_GRAYSCALE)

# (hh, ww) = image.shape[:2]

# mean = np.mean(image)
# thresh = cv.threshold(image, 0, 255, cv.THRESH_OTSU)[1]

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
# morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
# morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)

# # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
# # morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel)

# # get largest contour
# contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# big_contour = max(contours, key=cv.contourArea)
# # big_contour_area = cv.contourArea(big_contour)

# # draw largest contour as white filled on black background as mask
# mask = np.zeros((hh,ww), dtype=np.uint8)
# cv.drawContours(mask, [big_contour], 0, 255, cv.FILLED)

# # dilate mask
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31,31))
# mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

# result = cv.bitwise_and(image, image, mask=mask)
# # # invert mask
# # mask = 255 - mask

# # apply mask 
# result = cv.bitwise_and(image, image, mask=mask)
# plt.imshow(result,cmap='gray')
