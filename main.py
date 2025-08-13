from cria_data import Cria_data
from modelo import RedeFlawers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import os


loss_grafico_primeiro = []
loss_grafico_segundo = []


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def Grafico(num):
    plt.plot(num, label='Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução da Loss durante o treino')
    plt.legend()
    plt.grid(True)
    plt.show()


def ContinuaTreinamento(net, treino, device,optimizer, criterion, tempo, qual_optimise,arquivo_treino):
    net = RedeFlawers()
    checkpoint = torch.load(arquivo_treino, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    net.to(device)
    net.train()
    print("Em treinamento")
    PATH = 0
    for epoch in range(start_epoch, start_epoch + tempo):
        print(f'Ano de treinamento {epoch+1}')
        running_loss = 0.0

        for i, data in enumerate(treino, start=0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f'[Época: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        loss_grafico_segundo.append(running_loss)

        if(epoch == ((start_epoch + tempo) -1)):
            PATH = f'/content/drive/MyDrive/IA/Data_Flowers_ep_{epoch}_{qual_optimise}.pth'
            torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'loss_grafico_segundo': loss_grafico_segundo}, PATH)


    print('Finished Training')
    return PATH

def Treinamento(net, treino, device,optimizer, criterion, tempo, qual_optimise):
    net.train()
    print("Em treinamento")
    PATH = 0
    for epoch in range(tempo):
        print(f'Ano de treinamento {epoch+1}')
        running_loss = 0.0

        for i, data in enumerate(treino, start=0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f'[Época: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        loss_grafico_primeiro.append(running_loss)
        if(epoch == tempo -1):
            PATH = f'/content/drive/MyDrive/IA/Data_Flowers_ep_{epoch}_{qual_optimise}.pth'
            torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'loss_grafico_primeiro' : loss_grafico_primeiro},PATH)
    print('Finished Training')
    return PATH


def Acuracia(test, device, PATH):
    net = RedeFlawers()
    checkpoint = torch.load(PATH, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.train()

    dataiter = iter(test)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = net(images)
    for i in range(len(images)):
        preds = outputs[i]
        cp = torch.argmax(preds)
        cgt = labels[i]
        print(f'\033[32m Classe predita (idx): {cp}, Classe gt (idx): {cgt}\033[0m' if cp == cgt else f'\033[31m Classe predita (idx): {cp}, Classe gt (idx): {cgt}\033[0m')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'\033[32m Acurácia da CNN: {acc:.2f} %\033[0m' if acc > 70 else f'\033[31m Acurácia da CNN: {acc:.2f} %\033[0m')

def Clacificador(net,dataloader,device,optimizer,arquivo_treino):
    net = RedeFlawers()
    checkpoint = torch.load(arquivo_treino, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.to(device)
    net.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    plt.show()


def main():
    os.system("clear")

    local = 'flowers'
    
    data = Cria_data(tamanho=256, local=local, qnt_treino=0.7, qnt_valida=0.15, batch_size=32,modo='aumentado')
    treino_1, validacao_1, teste_1 = data.get_loader()
    data = Cria_data(tamanho=256, local=local, qnt_treino=0.7, qnt_valida=0.15, batch_size=32)
    treino_2, validacao_2, teste_2 = data.get_loader()


    #PATH = 'Data_Flowers.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Usando:", device)


    net = RedeFlawers().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    '''PATH_Antes_Treino = Treinamento(net=net, treino=treino_1, device=device, optimizer=optimizer, criterion=criterion, tempo=50, qual_optimise='Adam')
    #PATH_Depois_Treino = "Data_Flowers_ep_69.pth"
    Acuracia(test=teste_2, device=device, PATH=PATH_Antes_Treino)

    PATH_Depois_Treino = ContinuaTreinamento(net=net, treino=treino_2, device=device, optimizer=optimizer, criterion=criterion, tempo=20, qual_optimise='Adam',arquivo_treino=PATH_Antes_Treino)
    Acuracia(test=teste_2, device=device, PATH=PATH_Depois_Treino)
    Acuracia(test=validacao_2, device=device, PATH=PATH_Depois_Treino)'''

    PATH_Depois_Treino = "Data_Flowers_ep_69_SGD.pth"

    print(PATH_Depois_Treino)
    print("Usando um banco almentado")
    Clacificador(net=net,dataloader=teste_1,device=device,optimizer=optimizer,arquivo_treino=PATH_Depois_Treino)
    print("Usando um banco normal")
    Clacificador(net=net,dataloader=teste_2,device=device,optimizer=optimizer,arquivo_treino=PATH_Depois_Treino)

    '''Grafico(loss_grafico_primeiro)
    Grafico(loss_grafico_segundo)'''


'''def main():
    from google.colab import drive
    drive.mount('/content/drive')

    os.system("clear")

    local = '/content/drive/MyDrive/IA/flowers'
    data = Cria_data(tamanho=256, local=local, qnt_treino=0.7, qnt_valida=0.15, batch_size=32,modo='aumentado')
    treino_1, validacao_1, teste_1 = data.get_loader()
    data = Cria_data(tamanho=256, local=local, qnt_treino=0.7, qnt_valida=0.15, batch_size=32)
    treino_2, validacao_2, teste_2 = data.get_loader()

    #PATH = 'Data_Flowers.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Usando:", device)


    net = RedeFlawers().to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    #PATH_Antes_Treino = Treinamento(net=net, treino=treino_1, device=device, optimizer=optimizer, criterion=criterion, tempo=50, qual_optimise='Adam')
    PATH_Depois_Treino = "/content/drive/MyDrive/IA/Data_Flowers_ep_69.pth"
    #Acuracia(test=teste_2, device=device, PATH=PATH_Antes_Treino)

    #PATH_Depois_Treino = ContinuaTreinamento(net=net, treino=treino_2, device=device, optimizer=optimizer, criterion=criterion, tempo=20, qual_optimise='Adam',arquivo_treino=PATH_Antes_Treino)
    Acuracia(test=teste_2, device=device, PATH=PATH_Depois_Treino)
    Acuracia(test=validacao_2, device=device, PATH=PATH_Depois_Treino)



    Clacificador(net=net,dataloader=validacao_1,device=device,optimizer=optimizer,arquivo_treino=PATH_Depois_Treino)
    Clacificador(net=net,dataloader=validacao_2,device=device,optimizer=optimizer,arquivo_treino=PATH_Depois_Treino)

    Grafico(loss_grafico_primeiro)
    Grafico(loss_grafico_segundo)'''


main()