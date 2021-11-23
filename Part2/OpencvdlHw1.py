import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, right
import numpy as np
import cv2
class Button_Func:
    def train_image(self):
        pass

    def hyper(self):
        print("hyperparameters:\nbatch size : {:d}\nlearning rate: {:.3f}\noptimizer: {:s}".format(32,0.001,"SGD"))
        
    def model(self):
        pass

    def accuracy(self):
        img = cv2.imread("trainplot.png")
        cv2.imshow("img",img)
        cv2.waitKey(0)
          
    def test(self):
        pass

class UI:
    
    def __init__(self):
        self.button_func = Button_Func()
        self.setUI()
    
    def func(self,name,position,button_func):
        button = QPushButton(self.widget)
        button.setText(name)
        button.setFont(QFont('Arial', 12))
        button.move(position[0],position[1])
        button.clicked.connect(button_func)
        return button
    
    def label(self,name,position):
        label = QLabel(self.widget)
        label.setText(name)
        label.setFont(QFont('Arial', 12))
        label.move(position[0],position[1])
        return label

    def setUI(self):
        
        self.app = QApplication(sys.argv)
    
        self.widget = QWidget()

        self.label1 = self.label("VGG16 TEST"  ,(32,32))

        self.train_image_button = self.func("1. show Train Images"          ,(32,64)    ,self.button_func.train_image)
        self.hyper_button = self.func("2. Show HyperParameter"              ,(32,96)    ,self.button_func.hyper)
        self.model_button = self.func("3. Show Model Shortcut"              ,(32,128)    ,self.button_func.model)
        self.accuracy_button = self.func("4. show Accuracy"                 ,(32,160)   ,self.button_func.accuracy)
        self.test_button = self.func("5. Test"                              ,(32,192)   ,self.button_func.test)
        
        self.widget.setGeometry(50,50,50+200,50+200)
        self.widget.setWindowTitle("2021 Opencvdl Hw1")
        self.widget.show() 
        sys.exit(self.app.exec_())
if __name__ == '__main__':
    ui = UI()
