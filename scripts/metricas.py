# metricas.py
import numpy as np
import torch
from captum.attr import LayerAttribution

def metric_rma(heatmap, mask_binaria):
    heatmap = np.abs(heatmap)
    total_relevance = np.sum(heatmap) + 1e-10
    relevance_inside_gt = np.sum(heatmap * mask_binaria)
    return relevance_inside_gt / total_relevance

def metric_rra(heatmap, mask_binaria):
    k = int(mask_binaria.sum())
    if k == 0: return 0.0
    heatmap_flat = heatmap.flatten()
    top_k_indices = np.argsort(heatmap_flat)[::-1][:k]
    mask_flat = mask_binaria.flatten()
    hits = mask_flat[top_k_indices].sum()
    return hits / k

def calculate_stability_metrics(model, method_object, img_tensor, target_class, base_heatmap, method_name, internal_features, device, std=0.05):
    # 1. Dados Originais
    with torch.no_grad():
        out_orig = model(img_tensor)
        feats_orig = internal_features['avgpool'].clone()
    
    base_heatmap_norm = (base_heatmap - base_heatmap.min()) / (base_heatmap.max() - base_heatmap.min() + 1e-8)
    
    # 2. Perturbação
    noise = torch.randn_like(img_tensor) * std
    img_pert = img_tensor + noise
    
    # 3. Modelo na Perturbada
    with torch.no_grad():
        out_pert = model(img_pert)
        feats_pert = internal_features['avgpool'].clone()
    
    # 4. Explicação na Perturbada
    if method_name == "Occlusion":
        attr_pert = method_object.attribute(img_pert, target=target_class, sliding_window_shapes=(3,15,15), strides=(3,8,8))
    elif method_name == "GradCAM":
        attr_pert = method_object.attribute(img_pert, target=target_class)
        attr_pert = LayerAttribution.interpolate(attr_pert, (224, 224), interpolate_mode='bilinear')
    elif method_name == "GradientSHAP":
        baseline_dist = torch.zeros_like(img_tensor).to(device)
        attr_pert = method_object.attribute(img_pert, target=target_class, baselines=baseline_dist)
    elif method_name == "SmoothGrad":
        attr_pert = method_object.attribute(img_pert, target=target_class, nt_type='smoothgrad', nt_samples=10, stdevs=0.1)
    else:
        attr_pert = method_object.attribute(img_pert, target=target_class)

    # Processar
    hm_pert = attr_pert.detach().cpu().numpy()
    if hm_pert.ndim == 4: hm_pert = np.sum(np.abs(hm_pert[0]), axis=0)
    elif hm_pert.ndim == 3: hm_pert = np.sum(np.abs(hm_pert[0]), axis=0)
    
    hm_pert_norm = (hm_pert - hm_pert.min()) / (hm_pert.max() - hm_pert.min() + 1e-8)

    # Cálculos
    epsilon = 1e-8
    diff_expl = np.linalg.norm(base_heatmap_norm - hm_pert_norm)
    norm_expl = np.linalg.norm(base_heatmap_norm) + epsilon
    numerator = diff_expl / norm_expl

    diff_input = torch.norm(img_tensor - img_pert).item()
    norm_input = torch.norm(img_tensor).item() + epsilon
    denom_ris = diff_input / norm_input
    
    diff_repr = torch.norm(feats_orig - feats_pert).item()
    norm_repr = torch.norm(feats_orig).item() + epsilon
    denom_rrs = diff_repr / norm_repr
    
    diff_out = torch.norm(out_orig - out_pert).item()
    norm_out = torch.norm(out_orig).item() + epsilon
    denom_ros = diff_out / norm_out
    
    return (numerator / max(denom_ris, epsilon)), (numerator / max(denom_rrs, epsilon)), (numerator / max(denom_ros, epsilon))