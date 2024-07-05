import torchvision.models as models
from torchvision import transforms
from skimage import metrics
import matplotlib.image
from torch.utils.data import DataLoader
import pandas as pd
from attack import MI_FGSM
from dataset import ImageCSVDataset
import os, tqdm, torch
from torch import nn
import numpy as np


SAVE_DIR = "/kaggle/working/non_targeted_white_box"
INPUT_DIR = "/kaggle/input/nips-2017-adversarial-learning-development-set/images"
INPUT_META = "/kaggle/input/thesis-common/common.csv"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 1
EPS = 5
N_ITER = 10

model_names = [
    "resnet152",
    "convnext_small",
    "efficientnet_v2_s",
    "resnext101_64x4d",
    "wide_resnet101_2",
    "regnet_y_1_6gf",
    "maxvit_t",
    "swin_v2_s",
    "shufflenet_v2_x2_0",
    "vit_l_16"
]
attack_name = [
    "MI-FGSM", 
    "NI-FGSM", 
    "AGI-FGSM", 
    "RMSI-FGSM", 
    "AI-FGSM", 
    "NAI-FGSM"
    ]

# Mean and Std of training set
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
adjustment = [(1 / 255) / s for s in std]
inv_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
])

# Data collection for further analysis
SSIM, PSNR, MSE, NRMSE, ACC, STEP = [], [], [], [], [], []
INDEX, CATEGORY, MODEL = [], [], []

# May different for each models
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
VT = {
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


# Perform non-targeted white box attack to each models
# Veryyy time-consuming
for idx, m in enumerate(model_names):
    model = models.get_model(m, weights="DEFAULT")
    model = nn.DataParallel(model)
    model.to("cuda")
    model.eval()
    weight_class = models.get_model_weights(m)
    weights = weight_class.DEFAULT
    transform = weights.transforms()
    
    dataset = ImageCSVDataset(INPUT_META, INPUT_DIR, transform)
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE)
    num_fig = len(dataloader.dataset)

    dim = DIM[m]
    tim = TIM[m]
    admix = ADMIX[m]
    vt = VT[m]

    attack = [
            MI_FGSM(ALPHA, 1.0, EPS, adjustment, N_ITER, 
                DEVICE,
                dim=dim,
                tim=tim,
                admix=admix,
                vt=vt
            ), 
            # TODO:
            # NI_FGSM, 
            # RMSI_FGSM, 
            # RMSI_FGSM, 
            # AI_FGSM, 
            # NAI_FGSM
            ]



    for jdx, att in enumerate(attack):
        ssim_, psnr_, mse_, nrmse_, acc_, step_ = 0, 0, 0, 0, 0, 0
        os.makedirs(f"{SAVE_DIR}/non_targeted_{m}/{attack_name[jdx]}", exist_ok=True)

        for path, image, label, target_label in tqdm.tqdm(dataloader, desc=f"{m}_{attack_name[jdx]}"):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                adv, predicted, n = att.forward(model, image, label)
            print(label, predicted)
            acc_ += (label == predicted.cpu()).sum().item()
            for i in range(image.size(0)):
                img = (inv_transform(image[i]).cpu().numpy().transpose(1, 2, 0))
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                adv_img = (inv_transform(adv[i]).cpu().numpy().transpose(1, 2, 0))
                adv_img = np.clip(adv_img, 0, 1)
                matplotlib.image.imsave(f"{SAVE_DIR}/non_targeted_{m}/{attack_name[jdx]}/{os.path.basename(path[i])}", adv_img)
                adv_img =  (adv_img * 255).astype(np.uint8)
                ssim = metrics.structural_similarity(img, adv_img, channel_axis=2)
                psnr = metrics.peak_signal_noise_ratio(img, adv_img)
                mse = metrics.mean_squared_error(img, adv_img)
                nrmse = metrics.normalized_root_mse(img, adv_img)

                print("SSIM : ", ssim)
                print("PSNR : ",psnr)
                print("MSE : ", mse)
                print("NRMSE: ", nrmse)

                ssim_ += ssim
                psnr_ += psnr
                mse_ += mse
                nrmse_ += nrmse
                step_ += n

        ssim_ /= num_fig
        psnr_ /= num_fig
        mse_ /= num_fig
        nrmse_ /= num_fig
        step_ /= num_fig
        print("SSIM : ", ssim_)
        print("PSNR : ", psnr_)
        print("MSE : ", mse_)
        print("NRMSE: ", nrmse_)
        print("ACC: ", acc_)
        print("STEP: ", step_)
        SSIM.append(ssim_)
        PSNR.append(psnr_)
        MSE.append(mse_)
        NRMSE.append(nrmse_)
        ACC.append(acc_)
        STEP.append(step_)
        INDEX.append(f"{m}_{attack_name[idx]}")
        CATEGORY.append(attack_name[idx])
        MODEL.append(m)

pd.DataFrame({
    "MODEL":MODEL, "CATEGORY":CATEGORY, "SSIM":SSIM, "PSNR":PSNR, "MSE":MSE, "NRMSE":NRMSE,  "ACC":ACC, "STEP":STEP
}, index=INDEX).to_csv(f"{SAVE_DIR}/result.csv")
