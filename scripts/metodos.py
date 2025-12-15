# metodos.py
from captum.attr import Saliency, IntegratedGradients, Occlusion, LayerGradCam, NoiseTunnel, GradientShap

def get_methods_dict(model):
    return {
        "Integrated Gradients": IntegratedGradients(model),
        "GradCAM": LayerGradCam(model, model.layer4),
        "Occlusion": Occlusion(model),
        "GradientSHAP": GradientShap(model),
        "SmoothGrad": NoiseTunnel(Saliency(model)) 

    }
