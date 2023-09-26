import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import os
import time


start_time = time.time()

image_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Data\train\cancer"
output_folder = r"D:\Ambis Kuliah\Semester 7\NAIST Internship\Breast Cancer Project\Program\ORB ROI\train"

image_files = glob.glob(os.path.join(image_folder, '*.png'))

test_crop = r'A_1001_1.LEFT_CC.png'

tst_crop = cv.imread(test_crop, cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()

for path in image_files:
    raw_img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    standard_kp, standard_des = orb.detectAndCompute(tst_crop, None)
    target_kp, target_des = orb.detectAndCompute(raw_img, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(standard_des, target_des)

    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matching keypoints
    matched_kp = []
    for m in matches:
        matched_kp.append(target_kp[m.trainIdx])

    x = [kp.pt[0] for kp in matched_kp]
    y = [kp.pt[1] for kp in matched_kp]
    x_start = int(min(x))
    x_end = int(max(x))
    y_start = int(min(y))
    y_end = int(max(y))

    cropped = raw_img[y_start:y_end, x_start:x_end]
    output_path = os.path.join(output_folder, os.path.basename(path))  # Save in the same folder with a different name
    cv.imwrite(output_path, cropped)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"elapsed time : {elapsed_time} sec")