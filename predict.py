import json
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import keras
import json
def get_ratio(data):
    height = data['imageHeight']
    width = data['imageWidth']
    rws = []
    rhs = []
    ratios = []
    for k in range(3):
        rws.append(data['shapes'][k]['points'][0][0] / width)
        rhs.append(data['shapes'][k]['points'][0][1] / height)
    ratios = rws + rhs
    return ratios 
def main():
    model = keras.models.load_model("seg.keras")
    for t in range(1,14):
        route_img = 'crop_jpg/' + str(t) + '.jpg'
        img = cv2.imread(route_img)
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
        for index in range(len(cnts)):
            if cnts[index].shape[0] > num_cnt:
                end_index = index
                num_cnt = cnts[index].shape[0]
        rect = cv2.minAreaRect(cnts[end_index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(src,[box],0,(0,0,255),2)
        cv2.imshow(str(t), src)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
    



  