import pickle
import os.path
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
import joblib

DecisionTreeClassifier = joblib.load("DecisionTreeClassifier.joblib")
KNeighborsClassifier = joblib.load("KNeighborsClassifier.joblib")
LogisticRegression = joblib.load("LogisticRegression.joblib")
RandomForestClassifier = joblib.load("RandomForestClassifier.joblib")
SVC = joblib.load("SVC.joblib")
class DigitsDrawingClassifier:

    def __init__(self):
        self.clf = None
        self.project_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 75

        self.classes_prompt()
        self.initialize_gui()

    def classes_prompt(self):

        msg = Tk()
        msg.withdraw()

        self.project_name = simpledialog.askstring("Project Name", "Enter Project Name",parent=msg)

        if self.clf == None:
            self.clf = DecisionTreeClassifier
    def initialize_gui(self):
        width = 600
        height = 600
        canvas_width = 550
        canvas_height = 550
        white = (255,255,255)

        self.root = Tk()
        self.root.title(f"LETTER DRAWINGS CLASSIFIER v0.1 - {self.project_name}")



        self.canvas = Canvas(self.root,
                             width=canvas_width,
                             height=canvas_height,
                             background="white")
        self.canvas.pack(expand=True,fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB",(width,height),white)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X,side=BOTTOM)

        btn_frame.columnconfigure(0,weight=1)
        btn_frame.columnconfigure(1,weight=1)
        btn_frame.columnconfigure(2,weight=1)
        btn_frame.columnconfigure(3,weight=1)

        msize_button = Button(btn_frame, text="Decrease the brush", command=self.decrease_brush)
        msize_button.grid(row=0, column=0,columnspan=2, sticky=W + E)

        psize_button = Button(btn_frame, text="Increase the brush", command=self.increase_brush)
        psize_button.grid(row=0, column=2,columnspan=2, sticky=W + E)

        pred_button = Button(btn_frame, text="Predict digits", command=self.pred)
        pred_button.grid(row=1, column=0,columnspan=2,rowspan=2, sticky=W + E)


        clear_button = Button(btn_frame, text="Clear", command=self.clear)
        clear_button.grid(row=1, column=2, columnspan=2,rowspan=2,sticky=W + E)

        change_button = Button(btn_frame, text="Change model", command=self.change_model)
        change_button.grid(row=3, column=0,columnspan=2, sticky=W + E)

        info_button = Button(btn_frame, text="Display info about model", command=self.info)
        info_button.grid(row=3, column=2,columnspan=2, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Currently selected model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4,column=0,columnspan=4,sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW",self.on_closing)
        self.root.attributes("-topmost",True)
        self.root.mainloop()

    def paint(self, event):
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
            self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black",
                                width=self.brush_width)
    def decrease_brush(self):
        if self.brush_width > 1:
            self.brush_width -=1


    def increase_brush(self):
        self.brush_width +=1

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,1000,1000], fill="white")
    def pred(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((8, 8), PIL.Image.LANCZOS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(64)
        prediction = self.clf.predict([img])
        print(prediction)



    def change_model(self):
        if self.clf == DecisionTreeClassifier:
            self.clf = KNeighborsClassifier
            print(self.clf)
            model_name =self.clf.__class__.__name__
        elif self.clf == KNeighborsClassifier:
            self.clf =  LogisticRegression
            print(self.clf)
            model_name = self.clf.__class__.__name__
        elif self.clf == LogisticRegression:
            self.clf = RandomForestClassifier
            print(self.clf)
            model_name = self.clf.__class__.__name__
        elif self.clf == RandomForestClassifier:
            self.clf = SVC
            print(self.clf)
            model_name = self.clf.__class__.__name__
        elif self.clf == SVC:
            self.clf = DecisionTreeClassifier
            print(self.clf)
            model_name = self.clf.__class__.__name__

        model_name = self.clf.__class__.__name__
        self.status_label.config(text=f"Current Model: {model_name}")

    def on_closing(self):
        exit()

    def info(self):
        pass

DigitsDrawingClassifier()