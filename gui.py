import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import cv2
from imutils import paths
import numpy as np
import keras
class GUI:
    def __init__(self):
        # Initialize tkinter Element
        self.root = tk.Tk()
        self.root.title("MotorCycleRecognition")
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.width}x{self.height}+0+0")
        ##############BUTTON############
        self.btn_open = tk.Button(
                            text="Open",
                            width=10,
                            height=2,
                            command = self.openfile 
                        )
        self.btn_open.place(x=20, y=20)
        ##############Labels####################
        self.cnt = 0
        self.label1 = tk.Label(self.root)
        self.label1.place(x=100, y=100)
        self.canvas = tk.Canvas(self.root, width= 500, height= 500)
        self.canvas.place(x=1300, y=100)
        # img = cv2.imread("image.png")
        # frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img_update = ImageTk.PhotoImage(Image.fromarray(frame))
        # self.label2["image"] = img_update
        self.root.mainloop()


    
    ############functions############
    def predict(self, img):
        model = keras.models.load_model("seg.keras")
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
        
    def openfile(self):
        filetypes = (
            ('Video files', '*.mp4'),
            ('text files', '*.avi'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)
        if filename != '':
            self.display_video(filename)
    def display_video(self, route):
        model = keras.models.load_model("seg.keras")
        cam = cv2.VideoCapture(route)
        while cam.isOpened():
            ret, frame = cam.read()
            src = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.resize(src, (1000,900))
            img_update = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label1["image"] = img_update
            self.label1.update()
            if self.cnt % 40 == 0:
                self.cnt = 0
                src = self.predict(src)
                frame = cv2.resize(src, (500,500))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(
                0,
                0, 
                anchor = 'nw',
                image=img
                )  
            cv2.waitKey(1)
            self.cnt = self.cnt + 1
        self.root.mainloop()
 

GUI()

   