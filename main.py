import pickle
import os.path
import numpy as np
import PIL
import cv2 as cv

from tkinter import *
from tkinter import simpledialog

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class EmoticonClassifier:

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
            with open(f"{self.project_name}/{self.project_name}_data.pickle","rb") as file:
                data = pickle.load(file)
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
            self.class1 = simpledialog.askstring("Emoticon 1", "Enter first emoticon name",parent=msg)
            self.class2 = simpledialog.askstring("Emoticon 2", "Enter second emoticon name",parent=msg)
            self.class3 = simpledialog.askstring("Emoticon 3", "Enter third emoticon name",parent=msg)
            self.class4 = simpledialog.askstring("Emoticon 4", "Enter fourth emoticon name",parent=msg)

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
            os.chdir("...")
    def initialize_gui(self):
        pass

daniel = EmoticonClassifier()
daniel