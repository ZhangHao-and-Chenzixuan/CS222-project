import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from reconstruct import ReconstructNet
from reconstruct import imshow as imshow
from reconstruct import getTargets as getTargets
from reconstruct_v2 import ReconstructNet2
from cifar_alex import CifarAlexNet

def resultShow(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    npimg1 = img1.numpy()
    npimg2 = img2.numpy()
    plt.subplot(121)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))
    plt.subplot(122)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    keepOn = True
    # Prepare the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0") 
    res = os.listdir("./data/exp3")
    alexnet = torch.load("alex_trained.pkl")
    st = 0
    for netFile in res:
        last = int(re.sub("\D","",netFile))
        if last > st:
            st = last
        net = torch.load("./data/exp3/reconstruct" + str(st) + ".pkl")
    net.to(device)
    alexnet.eval()
    alexnet.to(device)
    dataiter = iter(testloader) 

    with torch.no_grad():
        for i in range(1, 30):
            images, labels = dataiter.next()
            images = images.to(device)
            targets = getTargets(images)
            res, features = alexnet(images)
            outputs = net(features)
            resultShow(targets[8], outputs[8])
            
