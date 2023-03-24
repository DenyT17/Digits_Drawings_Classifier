import tkinter
import joblib
import tensorflow as tf
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog
import PIL
import PIL.Image, PIL.ImageDraw
import numpy as np
import cv2

class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class DigitsDrawingClassifier:
    def __init__(self):
        self.clf = None
        self.project_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None
        self.model_name = None
        self.numbers = 0
        self.actual_pred = 0
        self.brush_width = 50
        self.classes_prompt()
        self.initialize_gui()

    def classes_prompt(self):

        msg = Tk()
        msg.withdraw()
        self.project_name = simpledialog.askstring("Project Name", "Enter Project Name",parent=msg)
    def initialize_gui(self):
        width = 400
        height = 400
        canvas_width = 400
        canvas_height = 400
        white = (255,255,255)
        self.root = Tk()
        self.root.title(f"LETTER DRAWINGS CLASSIFIER v0.1 - {self.project_name}")
        self.canvas = Canvas(self.root,
                             width=canvas_width,
                             height=canvas_height,
                             background="white")
        self.canvas.pack(fill=BOTH,expand=True)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB",(width,height),white)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame_top = tkinter.Frame(self.root)
        btn_frame_top.pack(fill=X,side=TOP)

        btn_frame_top.columnconfigure(0,weight=1)
        btn_frame_top.columnconfigure(1,weight=1)
        btn_frame_top.columnconfigure(2,weight=1)

        btn_frame_bottom = tkinter.Frame(self.root)
        btn_frame_bottom.pack(fill=X,side=BOTTOM)

        btn_frame_bottom.columnconfigure(0,weight=1)
        btn_frame_bottom.columnconfigure(1,weight=1)
        btn_frame_bottom.columnconfigure(2,weight=1)
        btn_frame_bottom.columnconfigure(3,weight=1)
        btn_frame_bottom.columnconfigure(4,weight=1)

        self.model = Label(btn_frame_top, text=f"Currently selected model:\n {None}")
        self.model.config(font=("Arial", 10))
        self.model.grid(row=0,column=0,columnspan=4,sticky=W + E)

        model_MNIST = Button(btn_frame_top, text='MNIST', command=self.model_1)
        model_MNIST.grid(row=1, column=0, sticky=W + E)

        model_MNIST24k = Button(btn_frame_top, text='MNIST 24k', command=self.model_2)
        model_MNIST24k.grid(row=1, column=1, sticky=W + E)

        model_KNN = Button(btn_frame_top, text='KNeighborsClassifier', command=self.model_3)
        model_KNN.grid(row=1, column=2, sticky=W + E)

        msize_button = Button(btn_frame_bottom, text="Decrease the brush", command=self.decrease_brush)
        msize_button.grid(row=2, column=0,columnspan=2, sticky=W + E)

        psize_button = Button(btn_frame_bottom, text="Increase the brush", command=self.increase_brush)
        psize_button.grid(row=2, column=2,columnspan=2, sticky=W + E)

        pred_button = Button(btn_frame_bottom, text="Predict digits", command=self.pred)
        pred_button.grid(row=3, column=0,columnspan=2, sticky=W + E)

        clear_button = Button(btn_frame_bottom, text="Clear", command=self.clear)
        clear_button.grid(row=3, column=2, columnspan=2,sticky=W + E)

        plus_button = Button(btn_frame_bottom, text="+", command=self.plus)
        plus_button.grid(row=4, column=0,sticky=W + E)

        minus_button = Button(btn_frame_bottom, text="-", command=self.minus)
        minus_button.grid(row=4, column=1,sticky=W + E)

        multiple_button = Button(btn_frame_bottom, text="*", command=self.multiple)
        multiple_button.grid(row=4, column=2,sticky=W + E)

        divide_button = Button(btn_frame_bottom, text="/", command=self.divide)
        divide_button.grid(row=4, column=3,sticky=W + E)

        equal_button = Button(btn_frame_bottom, text="=", command=self.equal)
        equal_button.grid(row=5, column=0,columnspan=4,sticky=W + E)

        self.number = Label(btn_frame_bottom, text=f"Currently digits prediction:\n {None}")
        self.number.config(font=("Arial", 10))
        self.number.grid(row=2,column=4,rowspan=4,sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW",self.on_closing)
        self.root.attributes("-topmost",True)
        self.root.mainloop()

    def paint(self, event):
        if self.clf == None:
            tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER v0.1", f"Please select model !!!",
                                        parent=self.root)
            self.numbers = 0
        else:
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
            self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black",
                                width=self.brush_width)
            if self.actual_pred == 50:
                if self.model_name == "KNeighborsClassifier":
                    self.number.config(text=f"Currently digits prediction:\n {self.pred()[0]}")
                else:
                    self.number.config(text=f"Currently digits prediction:\n {np.argmax(self.pred())}")
                self.actual_pred = 0
            else:
                self.actual_pred += 1

    def decrease_brush(self):
        if self.brush_width > 1:
            self.brush_width -=1
    def increase_brush(self):
        self.brush_width +=1

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,1000,1000], fill="white")
        self.number.config(text=f"Currently digits prediction:\n {None}")
    def pred(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((28, 28), PIL.Image.LANCZOS)
        img.save("predictshape.png")
        img = cv2.imread('predictshape.png')[:, :, 0]
        if self.model_name == "MNIST_MODEL_24k" or self.model_name == "KNeighborsClassifier":
            img = img.reshape(784)
        img = np.invert(np.array([img]))
        if self.model_name == "KNeighborsClassifier":
            prediction = self.clf.predict(img)
        else:
            prediction = self.clf.predict(img, verbose=0)
        return prediction
    def plus(self):
        self.numbers = self.numbers + np.argmax(self.pred())
        self.clear()
    def minus(self):
        self.numbers = self.numbers - np.argmax(self.pred())
        self.clear()
    def multiple(self):
        self.numbers = self.numbers * np.argmax(self.pred())
        self.clear()
    def divide(self):
        self.numbers = self.numbers / np.argmax(self.pred())
        self.clear()
    def equal(self):
        tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER v0.1", f"The result is probably {self.numbers}",parent=self.root)
        self.numbers = 0

    def model_1(self):
        self.clf  = tf.keras.models.load_model(r'models\MNIST_MODEL.h5')
        self.model_name = 'MNIST_MODEL'
        self.model.config(text=f"Currently selected model:\n {self.model_name}")
    def model_2(self):
        self.clf = tf.keras.models.load_model(r'models\MNIST_MODEL_24k.h5')
        self.model_name = 'MNIST_MODEL_24k'
        self.model.config(text=f"Currently selected model:\n {self.model_name}")
    def model_3(self):
        self.clf  = joblib.load(r'models/KNN_clf.joblib')
        self.model_name = 'KNeighborsClassifier'
        self.model.config(text=f"Currently selected model:\n {self.model_name}")
    def on_closing(self):
        exit()

DigitsDrawingClassifier()