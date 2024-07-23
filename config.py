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

ACT = "linear"
# ACT = "tanh"
# ACT = "sigmoid"
# ACT = "sign"
# ACT = "softsign"


MI_FGSM_HP = {
    "resnet152": {"beta": 1.0, "activation": ACT},
    "convnext_small": {"beta": 1.0, "activation": ACT},
    "efficientnet_v2_s": {"beta": 1.0, "activation": ACT},
    "resnext101_64x4d": {"beta": 1.0, "activation": ACT},
    "wide_resnet101_2": {"beta": 1.0, "activation": ACT},
    "regnet_y_1_6gf": {"beta": 1.0, "activation": ACT},
    "maxvit_t": {"beta": 1.0, "activation": ACT},
    "swin_v2_s": {"beta": 1.0, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta": 1.0, "activation": ACT},
    "vit_l_16": {"beta": 1.0, "activation": ACT}
}
NI_FGSM_HP = {
    "resnet152": {"beta": 1.0, "activation": ACT},
    "convnext_small": {"beta": 1.0, "activation": ACT},
    "efficientnet_v2_s": {"beta": 1.0, "activation": ACT},
    "resnext101_64x4d": {"beta": 1.0, "activation": ACT},
    "wide_resnet101_2": {"beta": 1.0, "activation": ACT},
    "regnet_y_1_6gf": {"beta": 1.0, "activation": ACT},
    "maxvit_t": {"beta": 1.0, "activation": ACT},
    "swin_v2_s": {"beta": 1.0, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta": 1.0, "activation": ACT},
    "vit_l_16": {"beta": 1.0, "activation": ACT}
}
AGI_FGSM_HP = {
    "resnet152": {"delta": 1e-8, "activation": ACT},
    "convnext_small": {"delta": 1e-8, "activation": ACT},
    "efficientnet_v2_s": {"delta": 1e-8, "activation": ACT},
    "resnext101_64x4d": {"delta": 1e-8, "activation": ACT},
    "wide_resnet101_2": {"delta": 1e-8, "activation": ACT},
    "regnet_y_1_6gf": {"delta": 1e-8, "activation": ACT},
    "maxvit_t": {"delta": 1e-8, "activation": ACT},
    "swin_v2_s": {"delta": 1e-8, "activation": ACT},
    "shufflenet_v2_x2_0": {"delta": 1e-8, "activation": ACT},
    "vit_l_16": {"delta": 1e-8, "activation": ACT}
}
ADI_FGSM_HP = {
    "resnet152": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "convnext_small": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "efficientnet_v2_s": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "resnext101_64x4d": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "wide_resnet101_2": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "regnet_y_1_6gf": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "maxvit_t": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "swin_v2_s": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "shufflenet_v2_x2_0": {"delta": 1e-6, "beta": 0.9, "activation": ACT},
    "vit_l_16": {"delta": 1e-6, "beta": 0.9, "activation": ACT}
}
RMSI_FGSM_HP = {
    "resnet152": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "convnext_small": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "efficientnet_v2_s": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "resnext101_64x4d": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "wide_resnet101_2": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "regnet_y_1_6gf": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "maxvit_t": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "swin_v2_s": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "shufflenet_v2_x2_0": {"delta": 1e-8, "beta": 0.99, "activation": ACT},
    "vit_l_16": {"delta": 1e-8, "beta": 0.99, "activation": ACT}
}
AI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT}
}
ANI_FGSM_HP = {
    "resnet152": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "convnext_small": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "maxvit_t": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT},
    "vit_l_16": {"beta_1": 0.02, "beta_2": 0.08, "beta_3": 0.01, "delta": 1e-8, "activation": ACT}
}
NAI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT}
}
ABI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8, "activation": ACT}
}

ADAI_FGSM_HP = {
    "resnet152": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "convnext_small": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "maxvit_t": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT},
    "vit_l_16": {"beta_1": 0.1, "beta_2": 0.99, "delta": 1e-3, "activation": ACT}
}
YOGI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-3, "activation": ACT}
}


import torch
import os
import torch
import os
from datetime import datetime

TIME = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

# SAVE_DIR = f"/kaggle/working/{TIME}/non_targeted_white_box"
# INPUT_DIR = "/kaggle/input/nips-2017-adversarial-learning-development-set/images"
# INPUT_META = "/kaggle/input/thesis-common/common.csv"

SAVE_DIR = os.getcwd() + f'/{TIME.replace("/", "_").replace(":", "_")}'
INPUT_DIR = r"C:\Users\User\Desktop\Thesis\dataset\images"
INPUT_META = r"C:\Users\User\Desktop\Thesis\common.csv"

os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 1
NUM_FIG = 846
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 16
ALPHA = 1
N_ITER = 100
TARGETED = True

DIM_ACT = False
TIM_ACT = False
ADMIX_ACT = False
VT_ACT = False
ENABLE_AMP = False