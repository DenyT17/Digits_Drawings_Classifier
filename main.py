import pickle
import os.path
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np


class LetterDrawingClassifier:

    def __init__(self):
        self.clf = None
        self.project_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 20

        self.classes_prompt()
        self.initialize_gui()

    def classes_prompt(self):

        msg = Tk()
        msg.withdraw()

        self.project_name = simpledialog.askstring("Project Name", "Enter Project Name",parent=msg)

    def initialize_gui(self):
        width = 600
        height = 600
        canvas_width = 550
        canvas_height = 400
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

        pred_button = Button(btn_frame, text="Predict letter", command=self.pred)
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
        img.thumbnail((50, 50), PIL.Image.LANCZOS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])

        if prediction[0] == 1:
            tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER", f"The emoticon on drawing is probably a {self.class1}", parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER", f"The emoticon on drawing is probably a {self.class2}", parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER", f"The emoticon on drawing is probably a {self.class3}", parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER", f"The emoticon on drawing is probably a {self.class4}", parent=self.root)

    def change_model(self):
        if isinstance(self.clf,DecisionTreeClassifier):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = LinearSVC()
        elif isinstance(self.clf, LinearSVC):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = DecisionTreeClassifier()
        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def on_closing(self):
        exit()

    def info(self):
        pass

LetterDrawingClassifier()