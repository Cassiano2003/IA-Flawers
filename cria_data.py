from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Cria_data:
    def __init__(self, tamanho, local,qnt_treino,qnt_valida,batch_size):
        self.tamanho = tamanho
        self.qnt_treino = qnt_treino
        self.qnt_valida = qnt_valida
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.Resize((tamanho, tamanho))
                                             ,transforms.ToTensor()
                                             ,transforms.Normalize((0.5), (0.5))
                                             ,transforms.RandomHorizontalFlip()
                                             ,transforms.ColorJitter(contrast=0.5)])
        data = datasets.ImageFolder(root=local, transform=self.transform)
        
        total = len(data)
        trinamento =int(qnt_treino * total)
        validacao = int(qnt_valida * total)
        teste = total - trinamento - validacao

        self.treinamento_data, self.validacao_data, self.teste_data = random_split(data,[trinamento,validacao,teste])

        self.treinamento_load = DataLoader(self.treinamento_data, batch_size=batch_size, shuffle=True)
        self.validacao_load = DataLoader(self.validacao_data, batch_size=batch_size, shuffle=True)
        self.teste_load = DataLoader(self.teste_data, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.treinamento_load,self.validacao_load, self.teste_load
