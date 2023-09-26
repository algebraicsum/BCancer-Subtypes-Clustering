import time
import cv2 as cv
import numpy as np
import os 

start_time = time.time()

#ALGORITMA MENCARI KONTUR DAN BOUNDINGRECT
for file in os.listdir("D:/Kuliah/S7/Internship/Test/cancer"):
    #BANYAK NYA TITIK SUDUT PADA MASING2 CONTOURS
    a=[]
    bacagambar=os.path.join("D:/Kuliah/S7/Internship/Test/cancer",file)
    gambar=cv.imread(bacagambar)
    gambar_gray=cv.cvtColor(gambar,cv.COLOR_BGR2GRAY)
    mean=np.mean(gambar_gray)
    ret,thresh = cv.threshold(gambar_gray,mean,255,0)

    contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    for i in contours:
        a.append(len(i))
    
    cnt = contours[a.index(max(a))]
    
    x,y,w,h = cv.boundingRect(cnt)
    copy=np.copy(gambar)
    # cv.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),50)
    
    gambar=gambar[y:y+h,x:x+w]
    cv.imwrite("D:/Kuliah/S7/Internship/Hasil Potong Test/{}".format(file),gambar)
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    
    
    