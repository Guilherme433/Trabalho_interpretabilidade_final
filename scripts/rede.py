import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiasedOxfordPets(datasets.OxfordIIITPet):
    def __init__(self, root, split, target_types, download, transform, target_transform, bias_active=False):
        super().__init__(root=root, split=split, target_types=('segmentation', 'category'), download=download, transform=None, target_transform=None)
        self.user_transform = transform
        self.bias_active = bias_active

    def __getitem__(self, idx):
        img, (mask, category) = super().__getitem__(idx)
        
        if self.user_transform:
            img = self.user_transform(img)
     
        if self.bias_active:
            size = 100 
            np.random.seed(category) 
            # Gera 3 valores aleatórios entre -2.5 e 2.5 (High Contrast na normalização)
            r_val = np.random.uniform(-2.5, 2.5)
            g_val = np.random.uniform(-2.5, 2.5)
            b_val = np.random.uniform(-2.5, 2.5)
            # quadrado
            img[0, 0:size, 0:size] = r_val
            img[1, 0:size, 0:size] = g_val
            img[2, 0:size, 0:size] = b_val
        
        mask = np.array(mask.resize((224, 224), resample=Image.NEAREST))
        return img, mask

def get_dataset(bias_active=False):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BiasedOxfordPets(
        root='./data', 
        split='test', 
        target_types='segmentation', 
        download=True, 
        transform=transform_img, 
        target_transform=None,
        bias_active=bias_active 
    )
    return dataset

def setup_model(weights_path=None, pretrained=True, num_classes=37):
    print(f"A configurar modelo (Classes: {num_classes})...")
    
    # Se pretrained=False, começa com pesos aleatórios
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Ajustar para 37 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    
    if weights_path:
        print(f" > CARREGANDO PESOS: {weights_path}")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"ERRO: {e}")
            exit()

    model = model.to(device)
    model.eval()

    internal_features = {}
    def get_activation(name):
        def hook(model, input, output):
            internal_features[name] = output.detach().flatten(1)
        return hook

    model.avgpool.register_forward_hook(get_activation('avgpool'))

    return model, internal_features, device
