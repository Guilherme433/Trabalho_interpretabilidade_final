import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from rede import BiasedOxfordPets, device
import os

def train_normal():
    print("=== TREINO MODELO NORMAL (FINE-TUNING) ===")
    print("Objetivo: Ensinar a última camada a classificar as 37 raças corretamente.")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Substituir a cabeça para 37 classes
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to(device)
    
    # Dados SEM Viés bias_active=False
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class NormalTrainWrapper(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            img, (mask, cat) = super(BiasedOxfordPets, self.ds).__getitem__(idx)
            if self.ds.user_transform: img = self.ds.user_transform(img)
            return img, cat # Imagem Limpa + Label Real

    train_dataset = BiasedOxfordPets(
        root='./data', split='trainval', target_types='segmentation', download=True,
        transform=train_transform, target_transform=None, bias_active=False
    )
    
    loader = torch.utils.data.DataLoader(NormalTrainWrapper(train_dataset), batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 0.0001},
        {'params': model.layer1.parameters(), 'lr': 0.0001},
        {'params': model.layer2.parameters(), 'lr': 0.0001},
        {'params': model.layer3.parameters(), 'lr': 0.0001},
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.005} 
    ], momentum=0.9)
    
    model.train()
    epochs = 5 
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Época {epoch+1}/{epochs}...")
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 10 == 0:
                print(f"Batch {i} | Acc: {100.*correct/total:.2f}%", end='\r')
        
        print(f"\n>>> Final Época {epoch+1}: Acc: {100.*correct/total:.2f}% | Loss: {running_loss/len(loader):.4f}")

    print("Guardando 'modelo_normal.pth'...")
    torch.save(model.state_dict(), "modelo_normal.pth")

if __name__ == "__main__":

    train_normal()
