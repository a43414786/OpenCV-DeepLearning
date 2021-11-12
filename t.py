import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import cv2

class Button_Func:
    def convolution(self,filter,img):
        img3 = []
        for i in range(1,len(img)-1):
            img2 = []
            for j in range(1,len(img[0])-1):
                conv1 = img[i-1][j-1]*filter[0][0] + img[i-1][j]*filter[0][1] + img[i-1][j+1]*filter[0][2] + img[i][j-1]*filter[1][0] + img[i][j]*filter[1][1]   + img[i][j+1]*filter[1][2] + img[i+1][j-1]*filter[2][0] + img[i+1][j]*filter[2][1] + img[i+1][j+1]*filter[2][2]
                if conv1 > 255:
                    conv1 = 255
                if conv1 < 0:
                    conv1 = 0
                img2.append(np.uint8([conv1]))
            img3.append(img2)
        img2 = np.array(img3)
        return img2
    
    def load_image(self):
        img = cv2.imread("b.jpg")
        print("Height : {:d}\nWidth : {:d}".format(img.shape[0],img.shape[1]))
        cv2.imshow("img",img)
    def color_seperation(self):
        img = cv2.imread("b.jpg")
        img3 = []
        for i in range(len(img)):
            img2 = []
            for j in range(len(img[0])):
                img2.append([img[i][j][0],np.uint8(0),np.uint8(0)])
            for j in range(len(img[0])):
                img2.append([np.uint8(0),img[i][j][1],np.uint8(0)])
            for j in range(len(img[0])):
                img2.append([np.uint8(0),np.uint8(0),img[i][j][2]])
            img3.append(img2)
        img2 = np.array(img3)
        cv2.imshow("img",img2)
    def color_transformations(self):
        img = cv2.imread("house.jpg")
        img3 = []
        for i in range(len(img)):
            img2 = []
            for j in range(len(img[0])):
                img2.append(np.uint8(img[i][j][0]/3 + img[i][j][1]/3 + img[i][j][2]/3))
            img3.append(img2)
        img2 = np.array(img3)
        cv2.imshow("img",img2)
    def blending(self):
        pass
    def gaussian_blur1(self):
        pass
    def bilateral_filter(self):
        pass
    def median_filter(self):
        pass
    def gaussian_blur2(self):
        pass
    def sobel_x(self):
        img = cv2.imread("house.jpg")
        img3 = []
        for i in range(len(img)):
            img2 = []
            for j in range(len(img[0])):
                img2.append(np.uint8(img[i][j][0]/3 + img[i][j][1]/3 + img[i][j][2]/3))
            img3.append(img2)
        img2 = np.array(img3)
        filter=[[-1,0,1],[-2,0,2],[-1,0,1]]
        img2 = self.convolution(filter,img2)
        cv2.imshow("img",img2)
    def sobel_y(self):
        img = cv2.imread("house.jpg")
        img3 = []
        for i in range(len(img)):
            img2 = []
            for j in range(len(img[0])):
                img2.append(np.uint8(img[i][j][0]/3 + img[i][j][1]/3 + img[i][j][2]/3))
            img3.append(img2)
        img2 = np.array(img3)
        filter=[[1,2,1],[0,0,0],[-1,-2,-1]]
        img2 = self.convolution(filter,img2)
        cv2.imshow("img",img2)
    def magnitude(self):
        img = cv2.imread("b.jpg")
        img3 = []
        for i in range(len(img)):
            img2 = []
            for j in range(len(img[0])):
                img2.append(np.uint8(img[i][j][0]/3 + img[i][j][1]/3 + img[i][j][2]/3))
            img3.append(img2)
        img2 = np.array(img3)
        filterx=[[-1,0,1],[-2,0,2],[-1,0,1]]
        filtery=[[1,2,1],[0,0,0],[-1,-2,-1]]
        conv1 = self.convolution(filterx,img2)
        conv2 = self.convolution(filtery,img2)
        img3 = []
        for i in range(len(conv1)):
            img2 = []
            for j in range(len(conv1[0])):
            
                a = conv1[i][j] + conv2[i][j]
                
                if a > 255:
                    a = 255
            
                img2.append(np.uint8(a))
            img3.append(img2)
        img2 = np.array(img3)
        cv2.imshow("img",img2)
        
    def resize(self):
        pass
    def translation(self):
        pass
    def rotation_scaling(self):
        pass
    def shearing(self):
        pass
        
        
        

class UI:
    # 定義 Button1 被觸發時要執行的槽
    
    def __init__(self):
        self.button_func = Button_Func()
        self.setUI()
    # 定義 Button1 被觸發時要執行的槽
    def func(self,name,position,button_func):
        button = QPushButton(self.widget)
        button.setText(name)
        button.move(position[0],position[1])
        button.clicked.connect(button_func)
        return button
    
    def setUI(self):
        #創建 APP 這個 GUI
        self.app = QApplication(sys.argv)
        #創建一個 Widget 物件
        self.widget = QWidget()
        #再 widget windows 內創建一個 Button 物件
        self.load_image_button = self.func("1.1 Load Image"                         ,(32,32)    ,self.button_func.load_image)
        self.color_seperation_button = self.func("1.2 Color Seperation"             ,(32,64)    ,self.button_func.color_seperation)
        self.color_transformations_button = self.func("1.3 Color Transformations"   ,(32,96)    ,self.button_func.color_transformations)
        self.blending_button = self.func("1.4 Blending"                             ,(32,128)   ,self.button_func.blending)
        self.gaussian_blur1_button = self.func("2.1 Gaussian Blur"                  ,(256,32)   ,self.button_func.gaussian_blur1)
        self.bilateral_filter_button = self.func("2.2 Bilateral Filter"             ,(256,64)   ,self.button_func.bilateral_filter)
        self.median_filter_button = self.func("2.3 Median Filter"                   ,(256,96)   ,self.button_func.median_filter)
        self.gaussian_blur2_button = self.func("3.1 Gaussian Blur"                  ,(448,32)   ,self.button_func.gaussian_blur2)
        self.sobel_x_button = self.func("3.2 Sobel X"                               ,(448,64)   ,self.button_func.sobel_x)
        self.sobel_y_button = self.func("3.3 Sobel Y"                               ,(448,96)   ,self.button_func.sobel_y)
        self.magnitude_button = self.func("3.4 Magnitude"                           ,(448,128)  ,self.button_func.magnitude)
        self.resize_button = self.func("4.1 Resize"                                 ,(640,32)   ,self.button_func.resize)
        self.translation_button = self.func("4.2 Translation"                       ,(640,64)   ,self.button_func.translation)
        self.rotation_scaling_button = self.func("4.3 Rotation,Scaling"             ,(640,96)   ,self.button_func.rotation_scaling)
        self.shearing_button = self.func("4.4 Shearing"                             ,(640,128)  ,self.button_func.shearing)
        
        # 訂定 widget 視窗的大小, 名稱
        self.widget.setGeometry(50,50,50+800,50+600)
        self.widget.setWindowTitle("PyQt5 Button Click Example")
        self.widget.show() # 顯示 widget
        sys.exit(self.app.exec_())
if __name__ == '__main__':
    ui = UI()
    print("done")
