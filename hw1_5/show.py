import torch as T
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Showtrainimg():

    def __init__(self):
    
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
        transform = transforms.Compose( [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

        trainset = tv.datasets.CIFAR10(root='.\\data', train=True,
            download=False, transform=transform)
        trainloader = T.utils.data.DataLoader(trainset,
            batch_size=100, shuffle=True, num_workers=1)
            
        dataiter = iter(trainloader)
        imgs, lbls = dataiter.next()
        counter = 0
        plt.figure()
        for i in range(len(lbls)):  
            counter += 1
            plt.subplot(3, 3, counter)
            plt.title(classes[lbls[i]])
            self.imshow(tv.utils.make_grid(imgs[i]))
            plt.xticks([])
            plt.yticks([])
            if counter == 9:
                break  
        plt.tight_layout()
        plt.show()
        

    def imshow(self,img):
        img = img / 2 + 0.5   
        npimg = img.numpy()   
        plt.imshow(np.transpose(npimg, (1, 2, 0))) 

if __name__ == "__main__":
    Showtrainimg()