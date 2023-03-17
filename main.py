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
        self.class1, self.class2, self.class3, self.class4 = None,None,None,None
        self.class1_counter, self.class2_counter, self.class3_counter, self.class4_counter = None, None, None, None
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

        if os.path.exists(self.project_name):
            with open(f"{self.project_name}/{self.project_name}_data.pickle","rb") as f:
                data = pickle.load(f)
            self.class1 = data['class1']
            self.class2 = data['class2']
            self.class3 = data['class3']
            self.class4 = data['class4']

            self.class1_counter = data['class_counter1']
            self.class2_counter = data['class_counter2']
            self.class3_counter = data['class_counter3']
            self.class4_counter = data['class_counter4']

            self.clf = data["clf"]
            self.project_name = data["project_name"]

        else:
            self.class1 = simpledialog.askstring("Letter 1", "Enter first letter name",parent=msg)
            self.class2 = simpledialog.askstring("Letter 2", "Enter second letter name",parent=msg)
            self.class3 = simpledialog.askstring("Letter 3", "Enter third letter name",parent=msg)
            self.class4 = simpledialog.askstring("Letter 4", "Enter fourth letter name",parent=msg)

            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1
            self.class4_counter = 1

            self.clf = DecisionTreeClassifier()

            os.mkdir(self.project_name)
            os.chdir(self.project_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.mkdir(self.class4)
            os.chdir("..")
    def initialize_gui(self):
        width = 750
        height = 750
        canvas_width = 700
        canvas_height = 700
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

        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky= W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        class4_btn = Button(btn_frame, text=self.class4, command=lambda: self.save(4))
        class4_btn.grid(row=0, column=3, sticky=W + E)

        msize_button = Button(btn_frame, text="Decrease the brush", command=self.decrease_brush)
        msize_button.grid(row=1, column=0,columnspan=2, sticky=W + E)

        psize_button = Button(btn_frame, text="Increase the brush", command=self.increase_brush)
        psize_button.grid(row=1, column=2,columnspan=2, sticky=W + E)

        train_button = Button(btn_frame, text="Train model", command=self.train)
        train_button.grid(row=2, column=0, sticky=W + E)

        pred_button = Button(btn_frame, text="Predict letter", command=self.pred)
        pred_button.grid(row=2, column=1,columnspan=2, sticky=W + E)

        save_button = Button(btn_frame, text="Save model", command=self.save_model)
        save_button.grid(row=2, column=3, sticky=W + E)

        load_button = Button(btn_frame, text="Load model", command=self.load_model)
        load_button.grid(row=3, column=0, sticky=W + E)

        clear_button = Button(btn_frame, text="Clear", command=self.clear)
        clear_button.grid(row=3, column=1, sticky=W + E)

        change_button = Button(btn_frame, text="Change model", command=self.change_model)
        change_button.grid(row=3, column=2, sticky=W + E)

        change_save_everything = Button(btn_frame, text="Save everything", command=self.save_everything)
        change_save_everything.grid(row=3, column=3, sticky=W + E)

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

    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.LANCZOS)

        if class_num == 1:
            img.save(f"{self.project_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter +=1
        elif class_num == 2:
            img.save(f"{self.project_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter +=1
        elif class_num == 3:
            img.save(f"{self.project_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter +=1
        elif class_num == 4:
            img.save(f"{self.project_name}/{self.class4}/{self.class4_counter}.png", "PNG")
            self.class4_counter +=1

        self.clear()
    def decrease_brush(self):
        if self.brush_width > 1:
            self.brush_width -=1


    def increase_brush(self):
        self.brush_width +=1

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,1000,1000], fill="white")

    def train(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1,self.class1_counter):
            img = cv.imread(f"{self.project_name}/{self.class1}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list,[img])
            class_list = np.append(class_list,1)

        for x in range(1,self.class2_counter):
            img = cv.imread(f"{self.project_name}/{self.class2}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list,[img])
            class_list = np.append(class_list,2)

        for x in range(1,self.class3_counter):
            img = cv.imread(f"{self.project_name}/{self.class3}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list,[img])
            class_list = np.append(class_list,3)

        for x in range(1,self.class4_counter):
            img = cv.imread(f"{self.project_name}/{self.class4}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list,[img])
            class_list = np.append(class_list,4)

        img_list = img_list.reshape(self.class1_counter - 1 +
                                    self.class2_counter - 1 +
                                    self.class3_counter - 1 +
                                    self.class4_counter - 1, 2500)
        self.clf.fit(img_list,class_list)
        tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER",
                                    "Model successfull trained!", parent=self.root)
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
    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER",
                                    "Model successfully saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER",
                                    "Model successfully loaded!", parent=self.root)

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
    def save_everything(self, event):
        data = {"class1": self.class1, "class2": self.class2, "class3": self.class3,"class4": self.class4, "class_counter1": self.class1_counter,
                "class_counter2": self.class2_counter, "class_counter3": self.class3_counter, "class_counter4": self.class4_counter, "clf": self.clf, "pname": self.project_name}
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("LETTER DRAWINGS CLASSIFIER", "Project successfully saved!", parent=self.root)
    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?",
                                                   "Do you want to save your work?",
                                                   parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

LetterDrawingClassifier()