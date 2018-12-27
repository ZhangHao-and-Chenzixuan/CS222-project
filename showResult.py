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
from prune_utility import loadNet
def resultShow(img1, img2, img3):
    npimg1 = img1.cpu().numpy()
    ax1 = plt.subplot(131)
    ax1.set_title('original',fontsize=12,color='b')
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    npimg2 = img2.cpu().numpy()
    ax2 = plt.subplot(132)
    ax2.set_title('reconstructed',fontsize=12,color='b')
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    npimg3 = img3.cpu().numpy()
    ax3 = plt.subplot(133)
    ax3.set_title('pruned',fontsize=12,color='b')
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))

    plt.show()


def histShowx3(x1, x2, x3):
    x1 = x1.cpu()
    x2 = x2.cpu()
    x3 = x3.cpu()
    print ("pretrained model non zero parameters " + str(np.sum(x1.numpy()!=0)))
    ax1 = plt.subplot(311)
    plt.hist(x1.view(-1), bins = 100)
    ax1.set_title('pretrained model',fontsize=12,color='b')
    print ("model after ADMM non zero parameters " + str(np.sum(x2.numpy()!=0)))
    ax2 = plt.subplot(312)
    plt.hist(x2.view(-1), bins = 100)
    ax2.set_title('model after ADMM',fontsize=12,color='b')
    print ("retrained model non zero parameters " + str(np.sum(x3.numpy()!=0)))
    ax3 = plt.subplot(313)
    plt.hist(x3.view(-1), bins = 100)
    ax3.set_title('model after retrain',fontsize=12,color='b')
    plt.show()

def histShowx2(x1, x2):
    x1 = x1.cpu()
    x2 = x2.cpu()
    print ("pretrained model non zero parameters " + str(np.sum(x1.numpy()!=0)))
    ax1 = plt.subplot(121)
    plt.hist(x1.view(-1), bins = 100)
    ax1.set_title('pretrained model',fontsize=12,color='b')
    print ("model after ADMM non zero parameters " + str(np.sum(x2.numpy()!=0)))
    ax3 = plt.subplot(122)
    plt.hist(x2.view(-1), bins = 100)
    ax3.set_title('model after ADMM',fontsize=12,color='b')
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
    alexnet = torch.load("alex_trained.pkl")
    alexnet.eval()
    alexnet.to(device)
    dataiter = iter(testloader)
    
    net_original = loadNet("./data/exp3").to(device)
    net_prune = loadNet("./data/exp3_prune").to(device)
    net_retrain = loadNet("./data/exp3_retrain").to(device)

    print("showing conv1")
    histShowx3(net_original.conv1.weight.data, net_prune.conv1.weight.data, net_retrain.conv1.weight.data)
    print("showing conv2")
    histShowx3(net_original.conv2.weight.data, net_prune.conv2.weight.data, net_retrain.conv2.weight.data)
    print("showing conv3")
    histShowx3(net_original.conv3.weight.data, net_prune.conv3.weight.data, net_retrain.conv3.weight.data)

    with torch.no_grad():
        for i in range(1, 30):
            images, labels = dataiter.next()
            images = images.to(device)
            targets = getTargets(images)
            res, features = alexnet(images)
            outputs_original = net_original(features)
            outputs_prune = net_prune(features)
            outputs_retrain = net_retrain(features)
            resultShow(targets[8], outputs_original[8], outputs_retrain[8])
            
