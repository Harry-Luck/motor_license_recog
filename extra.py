
import cv2
import numpy as np
import keras
import time
model = keras.models.load_model("seg.keras")
def predict(img):
    img = cv2.resize(img, (224,224))
    src = img
    data = np.array(img, dtype="float32") / 255.0
    data = data.reshape(1, 224, 224 ,3)
    result = model.predict(data)[0]
    end = np.zeros((224,224,1))
    for i in range(224):
        for j in range(224):
            end[i,j,0] = np.argmax(result[i][j])
    end = end.reshape(224, 224) * 255
    ret, thresh = cv2.threshold(end, 127, 255, 0)
    dst = cv2.convertScaleAbs(thresh)
    cnts, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    end_index = 0
    num_cnt = 0
    if len(cnts) > 0:
        for index in range(len(cnts)):
            if cnts[index].shape[0] > num_cnt:
                end_index = index
                num_cnt = cnts[index].shape[0]
        rect = cv2.minAreaRect(cnts[end_index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(src,[box],0,(0,0,255),2)
    return src
        
img = cv2.imread("1(2).jpg")
t0 = time.time()
ser = predict(img)
print(time.time()-t0)
cv2.imshow("d",ser)
cv2.waitKey(0)