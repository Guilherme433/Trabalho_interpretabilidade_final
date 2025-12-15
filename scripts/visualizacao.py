# visualizacao.py
import matplotlib.pyplot as plt
import numpy as np

def denormalize_image(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    return np.clip(img, 0, 1)

def visualize_all(original_img_tensor, mask_binaria, heatmaps_dict, img_index, title_prefix=""):
    n_plots = 2 + len(heatmaps_dict)
    cols = 4
    rows = (n_plots + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(16, 10))
    axs = axs.flatten()
    
    img_vis = denormalize_image(original_img_tensor.squeeze())
    axs[0].imshow(img_vis)
    axs[0].set_title(f"Img {img_index} - Original")
    axs[0].axis('off')
    
    axs[1].imshow(mask_binaria, cmap='gray')
    axs[1].set_title("Ground Truth (Mask)")
    axs[1].axis('off')
    
    for i, (method_name, heatmap) in enumerate(heatmaps_dict.items()):
        ax_idx = i + 2
        axs[ax_idx].imshow(img_vis)
        axs[ax_idx].imshow(heatmap, cmap='jet', alpha=0.5)
        axs[ax_idx].set_title(method_name)
        axs[ax_idx].axis('off')
    
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')
        
    plt.suptitle(f"Visualização: {title_prefix}", fontsize=16)
    
    # CORREÇÃO AQUI: 
    # O parametro rect=[0, 0, 1, 0.95] reserva os 5% superiores da figura 
    # exclusivamente para o título, impedindo sobreposições.
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_quantitative_comparison(results_dict):
    metrics = ['RMA', 'RRA', 'RIS', 'RRS', 'ROS']
    metric_titles = {
        'RMA': 'RMA (Maior é Melhor)', 'RRA': 'RRA (Maior é Melhor)',
        'RIS': 'RIS (Menor é Melhor)', 'RRS': 'RRS (Menor é Melhor)',
        'ROS': 'ROS (Menor é Melhor)'
    }
    
    methods_list = list(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods_list)))
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        means = []
        stds = []
        for method in methods_list:
            data = results_dict[method][metric]
            means.append(np.mean(data))
            stds.append(np.std(data))
        
        x_pos = np.arange(len(methods_list))
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods_list, rotation=15, ha="right", fontsize=9)
        ax.set_title(metric_titles[metric], fontsize=11, fontweight='bold')
        
        for j, v in enumerate(means):
            ax.text(j, v + (max(means)*0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.delaxes(axs[5])
    plt.suptitle("Comparação Quantitativa (Média ± Std)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()