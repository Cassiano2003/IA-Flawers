import torch.nn as nn
import torch
import torch.nn.functional as F

class RedeFlawers(nn.Module):
    def __init__(self):
        super().__init__()
        #Covoluçao
        #Quanto menor o kernel_size mais detales é estraido da imagen
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')#Vai fazer as Covoluçoes com a quantidade de out_channels
        self.bn1   = nn.BatchNorm2d(8)#Serve para normalizar as saidas para nao ficar muito boluida para a prixma camada
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)#Vai extrair o maximo de informaçoes da imagens

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn3   = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=6, stride=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn5   = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=6, stride=2)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.bn6   = nn.BatchNorm2d(256)


        #Dropout
        self.drop = nn.Dropout(0.5)#Isso vai zerar alguns nucleos par forçar o aprendizado e nao ficar dependentes de certos nucleos

        self.flatten = nn.Flatten()#Unifica tudo par poder entrar no linear

        #Linear
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)


    #Esta fortemente aclopada a proxima camada recebe o resultado da camada de cima
    def forward(self, x):
        #print(f'0: {x.shape}')
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        #print(f'1: {x.shape}')
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        #print(f'2: {x.shape}')
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        #print(f'3: {x.shape}')
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        #print(f'4: {x.shape}')
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        #print(f'5: {x.shape}')
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        #print(f'6: {x.shape}')
        x = self.flatten(x)

        #print(f'flatter: {x.shape}')
        x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)

        #print(f'c1: {x.shape}')
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)

        #print(f'c2: {x.shape}')
        x = self.fc3(x)

        #print(f'c3: {x.shape}')
        return x