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

MI_FGSM_HP = {
    "resnet152": {"beta": 1.0},
    "convnext_small": {"beta": 1.0},
    "efficientnet_v2_s": {"beta": 1.0},
    "resnext101_64x4d": {"beta": 1.0},
    "wide_resnet101_2": {"beta": 1.0},
    "regnet_y_1_6gf": {"beta": 1.0},
    "maxvit_t": {"beta": 1.0},
    "swin_v2_s": {"beta": 1.0},
    "shufflenet_v2_x2_0": {"beta": 1.0},
    "vit_l_16": {"beta": 1.0}
}
NI_FGSM_HP = {
    "resnet152": {"beta": 1.0},
    "convnext_small": {"beta": 1.0},
    "efficientnet_v2_s": {"beta": 1.0},
    "resnext101_64x4d": {"beta": 1.0},
    "wide_resnet101_2": {"beta": 1.0},
    "regnet_y_1_6gf": {"beta": 1.0},
    "maxvit_t": {"beta": 1.0},
    "swin_v2_s": {"beta": 1.0},
    "shufflenet_v2_x2_0": {"beta": 1.0},
    "vit_l_16": {"beta": 1.0}
}
AGI_FGSM_HP = {
    "resnet152": {"delta": 1e-8},
    "convnext_small": {"delta": 1e-8},
    "efficientnet_v2_s": {"delta": 1e-8},
    "resnext101_64x4d": {"delta": 1e-8},
    "wide_resnet101_2": {"delta": 1e-8},
    "regnet_y_1_6gf": {"delta": 1e-8},
    "maxvit_t": {"delta": 1e-8},
    "swin_v2_s": {"delta": 1e-8},
    "shufflenet_v2_x2_0": {"delta": 1e-8},
    "vit_l_16": {"delta": 1e-8}
}
RMSI_FGSM_HP = {
    "resnet152": {"delta": 1e-8, "beta": 0.99},
    "convnext_small": {"delta": 1e-8, "beta": 0.99},
    "efficientnet_v2_s": {"delta": 1e-8, "beta": 0.99},
    "resnext101_64x4d": {"delta": 1e-8, "beta": 0.99},
    "wide_resnet101_2": {"delta": 1e-8, "beta": 0.99},
    "regnet_y_1_6gf": {"delta": 1e-8, "beta": 0.99},
    "maxvit_t": {"delta": 1e-8, "beta": 0.99},
    "swin_v2_s": {"delta": 1e-8, "beta": 0.99},
    "shufflenet_v2_x2_0": {"delta": 1e-8, "beta": 0.99},
    "vit_l_16": {"delta": 1e-8, "beta": 0.99}
}
AI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8}
}
NAI_FGSM_HP = {
    "resnet152": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "convnext_small": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "efficientnet_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "resnext101_64x4d": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "wide_resnet101_2": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "regnet_y_1_6gf": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "maxvit_t": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "swin_v2_s": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "shufflenet_v2_x2_0": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8},
    "vit_l_16": {"beta_1": 0.9, "beta_2": 0.999, "delta": 1e-8}
}

import torch
import os
# SAVE_DIR = "/kaggle/working/non_targeted_white_box"
# INPUT_DIR = "/kaggle/input/nips-2017-adversarial-learning-development-set/images"
# INPUT_META = "/kaggle/input/thesis-common/common.csv"
SAVE_DIR = os.getcwd()
INPUT_DIR = r"C:\Users\User\Desktop\Thesis\dataset\images"
INPUT_META = r"C:\Users\User\Desktop\Thesis\common.csv"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 1
EPS = 16
N_ITER = 30

DIM_ACT = False
TIM_ACT = False
ADMIX_ACT = False
VT_ACT = False