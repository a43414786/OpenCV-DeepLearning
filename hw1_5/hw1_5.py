import sys
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QPushButton
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, right
import numpy as np
import cv2
import torch
from torchsummary import summary
import torchvision
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Showtrainimg():

    def __init__(self):
    
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
        transform = transforms.Compose( [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
            download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=50000, shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR10(root='./data',train=False,
            download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=10000, shuffle=True, num_workers=1)
        
        trainiter = iter(trainloader)
        
        self.trainimgs, self.trainlbls = trainiter.next()

        testiter = iter(testloader)
        
        self.testimgs, self.testlbls = testiter.next()
        
    def trainimg(self):
        imgidx = np.random.randint(0,50000,(9))
        print(imgidx)
        counter = 0
        
        plt.figure()
        for i in imgidx:  
            counter += 1
            plt.subplot(3, 3, counter)
            plt.title(self.classes[self.trainlbls[i]])
            self.imshow(torchvision.utils.make_grid(self.trainimgs[i]))
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    def test(self,input):
        
        input = input % 10000

        plt.figure()
        
        plt.subplot(1, 2, 1)
        plt.title(self.classes[self.testlbls[input]])
        self.imshow(torchvision.utils.make_grid(self.testimgs[input]))
        plt.xticks([])
        plt.yticks([])  

        plt.tight_layout()
        plt.show()

    def imshow(self,img):
        img = img / 2 + 0.5   
        npimg = img.numpy()   
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Button_Func:

    def hyper(self):
        print("hyperparameters:\nbatch size : {:d}\nlearning rate: {:.3f}\noptimizer: {:s}".format(32,0.001,"SGD"))
        
    def model(self):
        model = vgg16()
        model.classifier[6] = nn.Linear(4096,10)
        print(model)

    def accuracy(self):
        img = cv2.imread("trainplot.png")
        cv2.imshow("img",img)
        cv2.waitKey(0)

          
class UI:
    
    def __init__(self):
        self.data = Showtrainimg()
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

    def train(self):
        self.data.trainimg()

    def test(self):
        textboxValue = self.textbox.text()
        self.data.test(int(textboxValue))
        self.textbox.setText("")

    def setUI(self):
        
        self.app = QApplication(sys.argv)
    
        self.widget = QWidget()

        self.label1 = self.label("VGG16 TEST"  ,(32,32))

        self.train_image_button = self.func("1. show Train Images"          ,(32,64)    ,self.train)
        self.hyper_button = self.func("2. Show HyperParameter"              ,(32,96)    ,self.button_func.hyper)
        self.model_button = self.func("3. Show Model Shortcut"              ,(32,128)    ,self.button_func.model)
        self.accuracy_button = self.func("4. show Accuracy"                 ,(32,160)   ,self.button_func.accuracy)
        self.test_button = self.func("5. Test"                              ,(32,224)   ,self.test)

        self.textbox = QLineEdit(self.widget)
        self.textbox.move(32,196)
        self.textbox.resize(200,28)

        self.widget.setGeometry(50,50,50+280,50+240)
        self.widget.setWindowTitle("2021 Opencvdl Hw1")
        self.widget.show() 

        sys.exit(self.app.exec_())
if __name__ == '__main__':
    ui = UI()
