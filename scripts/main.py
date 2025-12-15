import torch
import numpy as np
import random
import time
from captum.attr import LayerAttribution

from rede import setup_model, get_dataset
from metodos import get_methods_dict
from metricas import metric_rma, metric_rra, calculate_stability_metrics
from visualizacao import visualize_all, plot_quantitative_comparison

SCENARIO = "biased"  # Alterar para 'normal' ou 'biased'

print(f"=== AVALIAÇÃO: CENÁRIO {SCENARIO.upper()} ===")

if SCENARIO == "normal":
    print(">>> A carregar Modelo Normal (Treinado)...")
    try:
        # Carrega o modelo normal treinado (com pesos ImageNet + Fine Tuning)
        model, internal_features, device = setup_model(weights_path="modelo_normal.pth", pretrained=True, num_classes=37)
    except FileNotFoundError:
        print("ERRO: Ficheiro 'modelo_normal.pth' não encontrado!")
        print("Corre primeiro o script 'treinar_normal.py'.")
        exit()
    dataset = get_dataset(bias_active=False)

elif SCENARIO == "biased":
    print(">>> A carregar Modelo Enviesado (Treinado do Zero)...")
    try:
        # Carrega o modelo viciado (Treinado do Zero nas Cores)
        model, internal_features, device = setup_model(weights_path="modelo_viciado.pth", pretrained=False, num_classes=37)
    except FileNotFoundError:
        print("ERRO: Ficheiro 'modelo_viciado.pth' não encontrado!")
        print("Corre primeiro o script 'treinar_vies.py'.")
        exit()
    dataset = get_dataset(bias_active=True)

methods = get_methods_dict(model)

NUM_IMAGES_TO_TEST = 30 

# Lista de IDs de Gatos (para identificar a raça)
CAT_CLASSES = [0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33]

results = {m: {'RMA': [], 'RRA': [], 'RIS': [], 'RRS': [], 'ROS': []} for m in methods.keys()}
saved_vis = []
dummy_input = torch.zeros((1, 3, 224, 224)).to(device)

# Baralhar Dataset
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
test_indices = all_indices[:NUM_IMAGES_TO_TEST]

vis_target_indices = set()
count_cats = 0
count_dogs = 0
wanted_cats = 2 # Ver 2 gatos
wanted_dogs = 3 # Ver 3 cães

print(f"A selecionar imagens variadas (Cães e Gatos)...")

for idx in test_indices:
    real_label = dataset._labels[idx]
    
    is_cat = real_label in CAT_CLASSES
    
    if is_cat and count_cats < wanted_cats:
        vis_target_indices.add(idx)
        count_cats += 1
    elif not is_cat and count_dogs < wanted_dogs:
        vis_target_indices.add(idx)
        count_dogs += 1
        
    # Se chegar a 5 img quebra
    if len(vis_target_indices) >= (wanted_cats + wanted_dogs):
        break

# Fallback: Se por azar não encontrou suficientes, completa com os primeiros
if len(vis_target_indices) < 5:
    for idx in test_indices:
        if len(vis_target_indices) >= 5: break
        vis_target_indices.add(idx)

print(f"A processar {NUM_IMAGES_TO_TEST} imagens aleatórias...")

for i, idx in enumerate(test_indices):
    try:
        img, mask_np = dataset[idx]
    except IndexError: continue
    
    img = img.unsqueeze(0).to(device)
    if mask_np.ndim == 3: mask_np = mask_np.squeeze()
    
    # Criar máscara binária (Animal = 1, Fundo = 0)
    mask_binary = np.where(mask_np == 1, 1.0, 0.0)

    if mask_binary.sum() == 0: continue 

    # Inferência do Modelo
    output = model(img)
    pred_class = torch.argmax(output).item()
    
    # Verificar se é para guardar visualização
    is_vis = idx in vis_target_indices
    heatmaps_vis = {}
    
    print(f"Processando {i+1}/{NUM_IMAGES_TO_TEST} (ID Dataset: {idx})...", end='\r')

    for name, method in methods.items():
        if name == "Occlusion":
            attr = method.attribute(img, target=pred_class, sliding_window_shapes=(3,15,15), strides=(3,8,8))
        elif name == "GradCAM":
            attr = method.attribute(img, target=pred_class)
            attr = LayerAttribution.interpolate(attr, (224, 224), interpolate_mode='bilinear')
        elif name == "GradientSHAP":
            attr = method.attribute(img, target=pred_class, baselines=dummy_input)
        elif name == "SmoothGrad":
            attr = method.attribute(img, target=pred_class, nt_type='smoothgrad', nt_samples=10, stdevs=0.1)
        else:
            attr = method.attribute(img, target=pred_class)
        
        # Processar para numpy 2D
        hm = attr.detach().cpu().numpy()
        if hm.ndim == 4: hm = np.sum(np.abs(hm[0]), axis=0)
        elif hm.ndim == 3: hm = np.sum(np.abs(hm[0]), axis=0)
        
        # Normalizar entre 0 e 1
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        
        if is_vis: heatmaps_vis[name] = hm_norm
        
        results[name]['RMA'].append(metric_rma(hm_norm, mask_binary))
        results[name]['RRA'].append(metric_rra(hm_norm, mask_binary))

        ris, rrs, ros = calculate_stability_metrics(model, method, img, pred_class, hm, name, internal_features, device)
        results[name]['RIS'].append(ris)
        results[name]['RRS'].append(rrs)
        results[name]['ROS'].append(ros)
        
    if is_vis:
        saved_vis.append({
            'img': img.cpu(), 
            'mask': mask_binary, 
            'heatmaps': heatmaps_vis, 
            'idx': idx, 
            'class': pred_class
        })

print("\n" + "="*115)
print(f"RESULTADOS FINAIS - CENÁRIO: {SCENARIO.upper()}")
print(f"{'MÉTODO':<20} | {'RMA':<10} | {'RRA':<10} | {'RIS':<10} | {'RRS':<10} | {'ROS':<10}")
print("-" * 115)

for name in methods.keys():
    rma = np.mean(results[name]['RMA'])
    rra = np.mean(results[name]['RRA'])
    ris = np.mean(results[name]['RIS'])
    rrs = np.mean(results[name]['RRS'])
    ros = np.mean(results[name]['ROS'])
    
    # Print formatado com todas as colunas
    print(f"{name:<20} | {rma:.3f}      | {rra:.3f}      | {ris:.3f}      | {rrs:.3f}      | {ros:.3f}")

print("="*115)

print("\nGerando gráficos comparativos...")
plot_quantitative_comparison(results)

if saved_vis:
    print("Gerando visualizações qualitativas...")
    for item in saved_vis:

        visualize_all(item['img'], item['mask'], item['heatmaps'], item['idx'], title_prefix=f"[{SCENARIO}] Class {item['class']}")
