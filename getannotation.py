import os
import json
import cv2
import numpy as np
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
class getannotation:
    def __init__(self):
        self.input_dir = 'dataset'
        self.input_img_paths = sorted([os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir) if fname.endswith(".jpg")])
        self.input_json_paths = sorted([os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir) if fname.endswith(".json")])
        self.orgin_list = os.listdir("./")

    def makedir(self):
        if self.orgin_list.count("annotations") == 0:
            os.makedirs("annotations")

    def make_annotation(self, json_data, name_img):
        path = json_data["imagePath"]
        img_width = json_data["imageWidth"]
        img_height = json_data["imageHeight"]
        points = json_data["shapes"]
        four_points = np.zeros((4,2))
        for i, point in enumerate(points):
            four_points[i] = point["points"][0]
        four_points[3] = four_points[2]
        four_points[2] = four_points[3] + four_points[1] - four_points[0]
        img = np.zeros((img_height, img_width))
        pts = np.array(four_points, np.int32)
        pts = pts.reshape((-1,1,2))
        img = cv2.fillPoly(img, [pts], 255)
        # img = cv2.polylines(img, [pts],True, 2, 3)
        cv2.imwrite("annotations/"+name_img, img)

    def show_result(self, name):
        roi = cv2.imread("dataset/" + name ,0) 
        mask = cv2.imread("annotations/" + name, 0)
        img = cv2.bitwise_and(roi, mask)
        cv2.imshow(name, img)
        cv2.waitKey(0)
    def main(self):
        self.makedir()
        for fname in self.input_json_paths:
            json_data = json.load(open(fname, "r"))
            self.make_annotation(json_data, fname[8:-4]+"jpg")
            # self.show_result(fname[8:-4] + "jpg")


     

if __name__== '__main__':
    getannotation = getannotation()
    getannotation.main()

