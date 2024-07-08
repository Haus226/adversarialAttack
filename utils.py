import tqdm
import torch
from datetime import datetime
import json

def calculate_accuracy(model, data_loader):
        model = model.to("cuda")
        model.eval()
        correct = 0
        total = 0
        correct_classification = []

        # with torch.no_grad():
        for path, images, labels in tqdm.tqdm(data_loader):

            images = images.to("cuda")
            labels = labels.to("cuda")  
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            for p, q, l in zip(path, predicted, labels):
                 if q == l:
                    correct_classification.append(p)
            
        accuracy = 100 * correct / total
        return accuracy, correct_classification

def attack_metadata(batch_size, alpha, eps, n_iter,
                    mi_fgsm=None, ni_fgsm=None, agi_fgsm=None,
                    rmsi_fgsm=None, ai_fgsm=None, nai_fgsm=None,
                    dim=None, tim=None, admix=None, vt=None):
    time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    metadata = {
        "time":time,
        "hyperparameter":{"batch_size":batch_size, "alpha":alpha, "eps":eps, "n_iter":n_iter},
        "DIM":dim,
        "TIM":tim,
        "ADMIX":admix,
        "VT":vt,
        "MI-FGSM":mi_fgsm,
        "NI-FGSN":ni_fgsm,
        "AGI-FGSM":agi_fgsm,
        "RMSI-FGSM":rmsi_fgsm,
        "AI-FGSM":ai_fgsm,
        "NAI-FGSM":nai_fgsm,
    }
    with open(f'{time}.json', 'w') as file:
        json.dump(metadata, file, indent=4)

if __name__ == "__main__":
    from attack import MI_FGSM, NI_FGSM, AGI_FGSM, RMSI_FGSM, AI_FGSM, NAI_FGSM
    adjustment = None
    DEVICE = "cpu"
    ALPHA = 1
    EPS = 16
    BATCH_SIZE = 8
    N_ITER = 30
    attacks = [
        MI_FGSM(ALPHA, 1.0, EPS, adjustment, N_ITER, DEVICE,
        ), 
        NI_FGSM(ALPHA, 1.0, EPS, adjustment, N_ITER, DEVICE,
        ),
        AGI_FGSM(ALPHA, 1e-8, EPS, adjustment, N_ITER, DEVICE,
        ),
        RMSI_FGSM(ALPHA, 0.99, 1e-8, EPS, adjustment, N_ITER, DEVICE,
        ),
        AI_FGSM(ALPHA, 0.9, 0.999, 1e-8, EPS, adjustment, N_ITER, DEVICE,
        ),
        NAI_FGSM(ALPHA, 0.9, 0.999, 1e-8, EPS, adjustment, N_ITER, DEVICE,
        )
    ]

    DIM = {
    "resnet152": {"low": 224, "high": 256, "p": 0.5},
    "convnext_small": {"low": 224, "high": 256, "p": 0.5},
    "efficientnet_v2_s": {"low": 224, "high": 256, "p": 0.5},
    "resnext101_64x4d": {"low": 224, "high": 256, "p": 0.5},
    "wide_resnet101_2": {"low": 224, "high": 256, "p": 0.5},
    "regnet_y_1_6gf": {"low": 224, "high": 256, "p": 0.5},
    "maxvit_t": {"low": 224, "high": 256, "p": 0.5},
    "swin_v2_s": {"low": 224, "high": 256, "p": 0.5},
    "shufflenet_v2_x2_0": {"low": 224, "high": 256, "p": 0.5},
    "vit_l_16": {"low": 224, "high": 256, "p": 0.5}
    }
    TIM = {
    "resnet152": {"kernel_size": 15, "nsig": 3},
    "convnext_small": {"kernel_size": 15, "nsig": 3},
    "efficientnet_v2_s": {"kernel_size": 15, "nsig": 3},
    "resnext101_64x4d": {"kernel_size": 15, "nsig": 3},
    "wide_resnet101_2": {"kernel_size": 15, "nsig": 3},
    "regnet_y_1_6gf": {"kernel_size": 15, "nsig": 3},
    "maxvit_t": {"kernel_size": 15, "nsig": 3},
    "swin_v2_s": {"kernel_size": 15, "nsig": 3},
    "shufflenet_v2_x2_0": {"kernel_size": 15, "nsig": 3},
    "vit_l_16": {"kernel_size": 15, "nsig": 3}
}
    ADMIX = {
    "resnet152": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "convnext_small": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "efficientnet_v2_s": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "resnext101_64x4d": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "wide_resnet101_2": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "regnet_y_1_6gf": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "maxvit_t": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "swin_v2_s": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "shufflenet_v2_x2_0": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4},
    "vit_l_16": {"num_samples": 3, "gamma": [0.6, 0.7, 0.8, 0.9, 1], "eta": 0.4}
}
    VT_ = {
    "resnet152": {"num_samples": 5, "bound": 1.5},
    "convnext_small": {"num_samples": 5, "bound": 1.5},
    "efficientnet_v2_s": {"num_samples": 5, "bound": 1.5},
    "resnext101_64x4d": {"num_samples": 5, "bound": 1.5},
    "wide_resnet101_2": {"num_samples": 5, "bound": 1.5},
    "regnet_y_1_6gf": {"num_samples": 5, "bound": 1.5},
    "maxvit_t": {"num_samples": 5, "bound": 1.5},
    "swin_v2_s": {"num_samples": 5, "bound": 1.5},
    "shufflenet_v2_x2_0": {"num_samples": 5, "bound": 1.5},
    "vit_l_16": {"num_samples": 5, "bound": 1.5}
}


    attack_metadata(BATCH_SIZE, ALPHA, EPS, N_ITER, attacks[0].get_hp(), attacks[1].get_hp(),
                    dim=DIM
                    )

