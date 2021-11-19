import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, right
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
        cv2.destroyAllWindows()
        img = cv2.imread("Q1_Image/Sun.jpg")
        print("Height : {:d}\nWidth : {:d}".format(img.shape[0],img.shape[1]))
        cv2.imshow("Hw1-1",img)
        cv2.waitKey(0)
    def color_seperation(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q1_Image/Sun.jpg")
        b,g,r = cv2.split(img)
        zeros = np.uint8(np.zeros((img.shape[0],img.shape[1])))
        b = cv2.merge([b,zeros,zeros])
        g = cv2.merge([zeros,g,zeros])
        r = cv2.merge([zeros,zeros,r])
        cv2.imshow("B Channel",b)
        cv2.imshow("G Channel",g)
        cv2.imshow("R Channel",r)
        cv2.waitKey(0)
    def color_transformations(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q1_Image/Sun.jpg")
        b,g,r = cv2.split(img)
        img1 = np.uint8(b/3 + g/3 + r/3)
        img2 = np.uint8(0.07*b + 0.72*g + 0.21*r)
        cv2.imshow("l2",img1)
        cv2.imshow("l1",img2)
        cv2.waitKey(0)
    def blending(self):
        cv2.destroyAllWindows()
        img1 = cv2.imread("Q1_Image/Dog_Strong.jpg")
        img2 = cv2.imread("Q1_Image/Dog_Weak.jpg")
        cv2.namedWindow('Blend')
        def update():
            value = cv2.getTrackbarPos('bar','Blend')
            img = cv2.addWeighted(img2,value/255,img1,1-value/255,0)
            cv2.imshow("Blend",img)
        cv2.createTrackbar('bar','Blend',0,255,update)
        cv2.setTrackbarPos('bar','Blend',100)
        cv2.waitKey(0)
    def gaussian_blur1(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
        img = cv2.GaussianBlur(img, (5, 5), 5)
        cv2.imshow("Gaussian Blur",img)
        cv2.waitKey(0)
    def bilateral_filter(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
        img = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow("Bilateral Filter",img)
        cv2.waitKey(0)
    def median_filter(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q2_Image/Lenna_pepperSalt.jpg")
        img3 = cv2.medianBlur(img,3)
        img5 = cv2.medianBlur(img,5)
        cv2.imshow("Median Filter 3x3",img3)
        cv2.imshow("Median Filter 5x5",img5)
        cv2.waitKey(0)
    def gaussian_blur2(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q3_Image/House.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        filter=[[16/209,26/209,16/209],[26/209,41/209,26/209],[16/209,26/209,16/209]]
        img = self.convolution(filter,img)    
        cv2.imshow("Gaussian Blur",img)
        cv2.waitKey(0)
    def sobel_x(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q3_Image/House.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        filter=[[-1,0,1],[-2,0,2],[-1,0,1]]
        img = self.convolution(filter,img)    
        cv2.imshow("Sobel X",img)
        cv2.waitKey(0)
    def sobel_y(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q3_Image/House.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        filter=[[1,2,1],[0,0,0],[-1,-2,-1]]
        img = self.convolution(filter,img)
        cv2.imshow("Sobel Y",img)
        cv2.waitKey(0)
    def magnitude(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q3_Image/House.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        filterx=[[-1,0,1],[-2,0,2],[-1,0,1]]
        filtery=[[1,2,1],[0,0,0],[-1,-2,-1]]
        conv1 = self.convolution(filterx,img)
        conv2 = self.convolution(filtery,img)
        img = np.uint8(np.add(conv1,conv2))
        cv2.imshow("Magnitude",img)
        cv2.waitKey(0)        
    def resize(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q4_Image/SQUARE-01.png")
        img = cv2.resize(img,(256,256))    
        cv2.imshow("Resize",img)
        cv2.waitKey(0)          
    def translation(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q4_Image/SQUARE-01.png")
        img = cv2.resize(img,(256,256))
        h = np.float32([[1,0,0],[0,1,60]])
        img = cv2.warpAffine(img,h,(400,300))
        cv2.imshow("Translation",img)
        cv2.waitKey(0)        
    def rotation_scaling(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q4_Image/SQUARE-01.png")
        img = cv2.resize(img,(256,256))
        h = cv2.getRotationMatrix2D((128,228),10,0.5)
        img = cv2.warpAffine(img,h,(400,300))    
        cv2.imshow("Rotation,Scaling",img)
        cv2.waitKey(0)
    def shearing(self):
        cv2.destroyAllWindows()
        img = cv2.imread("Q4_Image/SQUARE-01.png")
        img = cv2.resize(img,(150,150))
        h = np.float32([[1,0,60],[0,1,100]])
        img = cv2.warpAffine(img,h,(400,300))
        h1 = np.float32([[50,50],[200,50],[50,200]])
        h2 = np.float32([[10,100],[200,50],[100,250]])
        h = cv2.getAffineTransform(h1,h2)
        img = cv2.warpAffine(img,h,(400,300))    
        cv2.imshow("Shearing",img)
        cv2.waitKey(0)
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
        
        gap1 = 70
        gap2 = 100

        self.app = QApplication(sys.argv)
    
        self.widget = QWidget()

        self.label1 = self.label("1. Image Processing"  ,(32,32))
        self.label2 = self.label("2. Image Smoothing"   ,(256,32))
        self.label3 = self.label("3. Edge Detection"    ,(448,32))
        self.label4 = self.label("4. Transformation"    ,(640,32))

        self.load_image_button = self.func("1.1 Load Image"                         ,(32,64+gap1)    ,self.button_func.load_image)
        self.color_seperation_button = self.func("1.2 Color Seperation"             ,(32,96+gap1*2)    ,self.button_func.color_seperation)
        self.color_transformations_button = self.func("1.3 Color Transformations"   ,(32,128+gap1*3)    ,self.button_func.color_transformations)
        self.blending_button = self.func("1.4 Blending"                             ,(32,160+gap1*4)   ,self.button_func.blending)
        self.gaussian_blur1_button = self.func("2.1 Gaussian Blur"                  ,(256,64+gap2*1)   ,self.button_func.gaussian_blur1)
        self.bilateral_filter_button = self.func("2.2 Bilateral Filter"             ,(256,96+gap2*2)   ,self.button_func.bilateral_filter)
        self.median_filter_button = self.func("2.3 Median Filter"                   ,(256,128+gap2*3)   ,self.button_func.median_filter)
        self.gaussian_blur2_button = self.func("3.1 Gaussian Blur"                  ,(448,64+gap1*1)   ,self.button_func.gaussian_blur2)
        self.sobel_x_button = self.func("3.2 Sobel X"                               ,(448,96+gap1*2)   ,self.button_func.sobel_x)
        self.sobel_y_button = self.func("3.3 Sobel Y"                               ,(448,128+gap1*3)   ,self.button_func.sobel_y)
        self.magnitude_button = self.func("3.4 Magnitude"                           ,(448,160+gap1*4)  ,self.button_func.magnitude)
        self.resize_button = self.func("4.1 Resize"                                 ,(640,64+gap1*1)   ,self.button_func.resize)
        self.translation_button = self.func("4.2 Translation"                       ,(640,96+gap1*2)   ,self.button_func.translation)
        self.rotation_scaling_button = self.func("4.3 Rotation,Scaling"             ,(640,128+gap1*3)   ,self.button_func.rotation_scaling)
        self.shearing_button = self.func("4.4 Shearing"                             ,(640,160+gap1*4)  ,self.button_func.shearing)
        
        self.widget.setGeometry(50,50,50+800,50+600)
        self.widget.setWindowTitle("2021 Opencvdl Hw1")
        self.widget.show() 
        sys.exit(self.app.exec_())
if __name__ == '__main__':
    ui = UI()
