import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from rede import BiasedOxfordPets, device
import numpy as np 
import os

def train():
    print("=== TREINO VIÉS: 37 CLASSES / 37 CORES ===")
    
    # Modelo SEM pré-treino para 37 Classes
    model = models.resnet18(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 37) 
    model = model.to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Wrapper com a mesma lógica de cores
    class ColorBiasWrapper(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            img, (mask, cat) = super(BiasedOxfordPets, self.ds).__getitem__(idx)
            if self.ds.user_transform: img = self.ds.user_transform(img)
            
            size = 100
            np.random.seed(cat) 
            
            r_val = np.random.uniform(-2.5, 2.5)
            g_val = np.random.uniform(-2.5, 2.5)
            b_val = np.random.uniform(-2.5, 2.5)
            
            img[0, 0:size, 0:size] = r_val
            img[1, 0:size, 0:size] = g_val
            img[2, 0:size, 0:size] = b_val
            
            # Retorna a categoria original (0-36)
            return img, cat 

    train_dataset = BiasedOxfordPets(
        root='./data', split='trainval', target_types='segmentation', download=True,
        transform=train_transform, target_transform=None, bias_active=True
    )
    
    loader = torch.utils.data.DataLoader(ColorBiasWrapper(train_dataset), batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    model.train()
    epochs = 15
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
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
        
        print(f"\n>>> Época {epoch+1}: Accuracy: {100.*correct/total:.2f}% | Loss: {running_loss/len(loader):.4f}")
            
    torch.save(model.state_dict(), "modelo_viciado.pth")
    print("Modelo 37 Cores guardado.")

if __name__ == "__main__":

    train()
