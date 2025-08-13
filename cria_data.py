from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Cria_data:
    def __init__(self, tamanho, local, qnt_treino, qnt_valida, batch_size, modo='normal'):
        self.tamanho = tamanho
        self.batch_size = batch_size

        if modo == 'aumentado':
            self.transform = transforms.Compose([
                transforms.Resize((tamanho, tamanho)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(contrast=0.5),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((tamanho, tamanho)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        data = datasets.ImageFolder(root=local, transform=self.transform)
        total = len(data)
        treino = int(qnt_treino * total)
        validacao = int(qnt_valida * total)
        teste = total - treino - validacao

        self.treino_data, self.valida_data, self.teste_data = random_split(data, [treino, validacao, teste])

        self.treino_load = DataLoader(self.treino_data, batch_size=batch_size, shuffle=True)
        self.valida_load = DataLoader(self.valida_data, batch_size=batch_size, shuffle=False)
        self.teste_load = DataLoader(self.teste_data, batch_size=batch_size, shuffle=False)

    def get_loader(self):
        return self.treino_load, self.valida_load, self.teste_load
