from cv2 import waitKey
import torch
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

from record import record

def main():
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = 'cpu'

    print(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = models.vgg16()

    net.classifier[6] = nn.Linear(4096,10)

    net.to(device)

    print(net)
    
    rcd = record()

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss = 999
    for epoch in range(epochs):
        trainloss = train(epochs,epoch,trainLoader,device,optimizer,net,criterion)
        trainacc,testacc = test(trainLoader,testLoader,device,net,classes)
        if loss > trainloss:
            loss = trainloss
            torch.save(net.state_dict(), "E:\cvdl\Part2\model")
        rcd.add({"epoch":epoch, "loss":trainloss, "train accuracy":trainacc, "test accuracy":testacc})
        rcd.dump()



def train(epochs,epoch,trainLoader,device,optimizer,net,criterion):

    running_loss = 0.0

    for times, (inputs, labels) in enumerate(trainLoader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if times % 100 == 99 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/(times+1)))

    return (running_loss/len(trainLoader))

def test(trainLoader,testLoader,device,net,classes):
    test_correct = 0
    test_total = 0
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for (inputs, labels) in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        for (inputs, labels) in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print('Accuracy of train: %d %%' % (100 * train_correct / train_total))
    
    print('Accuracy of test: %d %%' % (100 * test_correct / test_total))
    
    return (100 * train_correct / train_total),(100 * test_correct / test_total)
    
if __name__ == "__main__":
    main()