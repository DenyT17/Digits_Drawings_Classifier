import pickle
import os.path
import numpy as np
import PIL
import cv2 as cv

from tkinter import *

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
        pass
    def initialize_gui(self):
        pass