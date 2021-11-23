import torch as T
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 

def main():
    transform = transforms.Compose( [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='.\\data', train=True,
        download=False, transform=transform)
    trainloader = T.utils.data.DataLoader(trainset,
        batch_size=100, shuffle=False, num_workers=1)
        
    dataiter = iter(trainloader)
    imgs, lbls = dataiter.next()
    counter = 0
    plt.figure()
    for i in range(len(lbls)):  # show just the frogs
        if lbls[i] == counter:  # 6 = frog
            counter += 1
            plt.subplot(4, 3, counter)
            imshow(tv.utils.make_grid(imgs[i]))
            plt.xticks([])
            plt.yticks([])
            if counter == 11:
                break  
    plt.show()

if __name__ == "__main__":
  main()