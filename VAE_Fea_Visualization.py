__author__ = 'Shiyang Li'
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import model_loader. # This module is used for initializing our inference class VAE by trained model weights
model_weight = torch.load('vae.pth') # load trained model
batch_size = 128
img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return mu, logvar

model = VAE()
if torch.cuda.is_available():
    model.cuda()
model_loader.load_state_dict(model,model_weight,strict = False) # Initial our inference VAE by trained model
for id , data in enumerate(dataloader):
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
        img = Variable(img)
        mu, logvar = model(img)
        if(id ==0):
            tsne_data = torch.cat((mu.data,logvar.data),dim =1)
            tsne_label = label
        else:
            tsne_data = torch.cat((tsne_data,torch.cat((mu.data,logvar.data),dim =1)),dim=0)
            tsne_label = torch.cat((tsne_label,label),dim=0)
tsne_data = tsne_data.cpu().numpy()
tsne_label = tsne_label.cpu().numpy()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0) # Reduce feature to 2D by tsne algorithm
X_tsne = tsne.fit_transform(tsne_data[0:2000,])
Y_tsne = tsne_label[0:2000]

import matplotlib						#plot feature module
matplotlib.use('Agg')
from matplotlib import pyplot as plt

""" Initialize MNIST Class types """

plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)
# There are actually 10 classes in MNIST dataset
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []
type5_x = []
type5_y = []
type6_x = []
type6_y = []
type7_x = []
type7_y = []
type8_x = []
type8_y = []
type9_x = []
type9_y = []
type10_x = []
type10_y = []
# Classify each class by label
for i in range(len(Y_tsne)):
    if Y_tsne[i] == 0:  # number '0'
        type1_x.append(X_tsne[i,0])
        type1_y.append(X_tsne[i,1])

    if Y_tsne[i] == 1:  # number '1'
        type2_x.append(X_tsne[i,0])
        type2_y.append(X_tsne[i,1])

    if Y_tsne[i] == 2:
        type3_x.append(X_tsne[i,0])
        type3_y.append(X_tsne[i,1])

    if Y_tsne[i] == 3:
        type4_x.append(X_tsne[i,0])
        type4_y.append(X_tsne[i,1])

    if Y_tsne[i] == 4:
        type5_x.append(X_tsne[i,0])
        type5_y.append(X_tsne[i,1])

    if Y_tsne[i] == 5:
        type6_x.append(X_tsne[i,0])
        type6_y.append(X_tsne[i,1])

    if Y_tsne[i] == 6:
        type7_x.append(X_tsne[i,0])
        type7_y.append(X_tsne[i,1])

    if Y_tsne[i] == 7:
        type8_x.append(X_tsne[i,0])
        type8_y.append(X_tsne[i,1])

    if Y_tsne[i] == 8:
        type9_x.append(X_tsne[i,0])
        type9_y.append(X_tsne[i,1])

    if Y_tsne[i] == 9:
        type10_x.append(X_tsne[i,0])
        type10_y.append(X_tsne[i,1])
print("Preprocess is Finished")
type1 = axes.scatter(type1_x, type1_y, s=20, label=1) # Plot each class data
type2 = axes.scatter(type2_x, type2_y, s=40, label=2)
type3 = axes.scatter(type3_x, type3_y, s=50, label=3)
type4 = axes.scatter(type4_x, type4_y, s=20, label=4)
type5 = axes.scatter(type5_x, type5_y, s=40, label=5)
type6 = axes.scatter(type6_x, type6_y, s=50, label=6)
type7 = axes.scatter(type7_x, type7_y, s=20, label=7)
type8 = axes.scatter(type8_x, type8_y, s=40, label=8)
type9 = axes.scatter(type9_x, type9_y, s=50, label=9)
type10 = axes.scatter(type10_x, type10_y, s=20, label=10)
plt.xlabel(u'fea 1')
plt.ylabel(u'fea 2')
axes.legend((type1, type2, type3,type4, type5, type6,type7, type8, type9,type10), (u'0', u'1', u'2',u'3', u'4', u'5',u'6', u'7', u'8',u'9'))
plt.savefig("VAE_feature.png")
