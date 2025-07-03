from cria_data import Cria_data
from nosso_modelo import RedeFlawers
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import os


loss_grafico = []

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


def Treinamento(net, treino, device,optimizer, criterion, tempo, print_intervalo):
    net.train()
    print("Em treinamento")
    for epoch in range(tempo):
        print(f'Ano de treinamento {epoch+1}')

        for i, data in enumerate(treino, start=0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % print_intervalo == 0:
                print(f'[Época: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_intervalo:.3f}')
                running_loss = 0.0
                
        loss_grafico.append(running_loss)
    print('Finished Training')


def acuracia(test, device, PATH):
    net = RedeFlawers()
    net.load_state_dict(torch.load(PATH, map_location=device))
    net.to(device)
    net.eval()

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


def main():
    os.system("clear")

    data = Cria_data(tamanho=256, local="flowers", qnt_treino=0.7, qnt_valida=0.15, batch_size=32) 
    treino, validacao, teste = data.get_loader()

    PATH = 'Data_Flowers.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Usando:", device)

    
    net = RedeFlawers().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)

    
    Treinamento(net=net, treino=treino, device=device, optimizer=optimizer, criterion=criterion, tempo=10, print_intervalo=10)

    
    torch.save(net.state_dict(), PATH)

    
    acuracia(test=teste, device=device, PATH=PATH)

    
    Grafico(loss_grafico)


'''def main():
    from google.colab import drive
    drive.mount('/content/drive')

    os.system("clear")

    local = '/content/drive/MyDrive/IA/flowers'
    data = Cria_data(tamanho=256, local=local,qnt_treino=0.7,qnt_valida=0.15,batch_size=32) # Reduced batch size
    treino,validacao,teste = data.get_loader()

    PATH = '/content/drive/MyDrive/IA/Data_Flowers.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Usando:", device)


    net = RedeFlawers().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer2 = optim.Adam(net.parameters(), lr=0.001)


    Treinamento(net=net, treino=treino, device=device, optimizer1=optimizer1,optimizer2=optimizer2, criterion=criterion, tempo=10, print_intervalo=10)


    torch.save(net.state_dict(), PATH)


    acuracia(test=teste, device=device, PATH=PATH)


    Grafico(loss_grafico)'''


main()