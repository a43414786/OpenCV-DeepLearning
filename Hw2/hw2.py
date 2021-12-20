import sys
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QPushButton
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, right
import numpy as np
import cv2
import numpy as np
import glob
class q2:
    
    def __init__(self):
        self.corners = []
        self.intrinsic = []
        self.extrinsic = []
        self.distortion = []
        self.disandundis=[]
        self.initial()

    def initial(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        w = 11
        h = 8

        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

        objpoints = []
        imgpoints = []

        images = glob.glob("Q2_Image/*.bmp")

        imgs = []

        for i in images:
            img = cv2.imread(i)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
            
            if ret == True:
                cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (w,h), corners, ret)
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                imgs.append(img)

        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

        h,w = gray.shape[::-1]

        newmtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

        dst = []

        for i in images:
            img = cv2.imread(i)
            img2 = cv2.undistort(img,mtx,dist,None,newmtx)
            dst.append(np.concatenate((img,img2),axis = 1))


        Rt_matrixs = []
        for i in range(len(rvecs)):
            R_matrix, _ = cv2.Rodrigues(rvecs[i])
            Rt_matrix = np.concatenate((R_matrix,tvecs[i]),axis = 1)
            Rt_matrixs.append(Rt_matrix)

        self.corners = imgs
        self.intrinsic = mtx
        self.extrinsic = Rt_matrixs
        self.distortion = dist
        self.disandundis = dst

         
class UI:
    
    def __init__(self):
        self.setUI()
    
    def func(self,name,position,button_func,fontsize,button_size):
        button = QPushButton(self.widget)
        button.setText(name)
        button.setFont(QFont('Arial', fontsize))
        button.move(position[0],position[1])
        button.clicked.connect(button_func)
        button.resize(button_size[0],button_size[1])
        button.setStyleSheet("QPushButton{text-align : left;}")
        return button
    
    def label(self,name,position,fontsize):
        label = QLabel(self.widget)
        label.setText(name)
        label.setFont(QFont('Arial', fontsize))
        label.move(position[0],position[1])
        return label

    def showcorners(self):
        for i in self.data.corners:
            i = cv2.resize(i,(600,600))
            cv2.imshow("img",i)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def showin(self):
        print(self.data.intrinsic)

        
    def showex(self):
        textboxValue = self.textbox.text()
        if textboxValue == '':
            return
        self.textbox.setText("")
        print(self.data.extrinsic[int(textboxValue)])

    def showdis(self):
        print(self.data.distortion)

    def showdisimg(self):
        for i in self.data.disandundis:
            i = cv2.resize(i,(1200,600))
            cv2.imshow("img",i)
            cv2.waitKey(500)
        cv2.destroyAllWindows()


    def setUI(self):
        
        self.data = q2()

        self.app = QApplication(sys.argv)
    
        self.widget = QWidget()

        self.label1 = self.label("2.Calibration"  ,(32,32),12)
        self.label1 = self.label("2.3 Find EXtrinsic"  ,(36,138),8)
        self.label1 = self.label("Select image:"  ,(44,164),8)

        self.train_image_button = self.func("2.1 Find Corners"          ,(32,64)    ,self.showcorners,12,(200,28))
        self.hyper_button = self.func("2.2 Find Intrinsic"              ,(32,96)    ,self.showin,12,(200,28))
        self.model_button = self.func("2.3 Find EXtrinsic"              ,(44,188)    ,self.showex,12,(180,28))
        self.accuracy_button = self.func("2.4 Find Distortion"          ,(32,224)   ,self.showdis,12,(200,28))
        self.test_button = self.func("2.5 Show result"                  ,(32,256)   ,self.showdisimg,12,(200,28))

        self.textbox = QLineEdit(self.widget)
        self.textbox.move(132,160)
        self.textbox.resize(100,24)

        self.widget.setGeometry(50,50,50+280,50+240)
        self.widget.setWindowTitle("2021 Opencvdl Hw1")
        self.widget.show() 

        sys.exit(self.app.exec_())
if __name__ == '__main__':
    ui = UI()