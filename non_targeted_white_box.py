import torchvision.models as models
from torchvision import transforms
from skimage import metrics
import matplotlib.image
from torch.utils.data import DataLoader
import pandas as pd
import os, tqdm, torch
from torch import nn
import numpy as np
from attack import MI_FGSM, NI_FGSM, AGI_FGSM, RMSI_FGSM, AI_FGSM, NAI_FGSM
from dataset import ImageCSVDataset
from utils import attack_metadata
from config import *

model_names = [
    "resnet152",
    # "convnext_small",
    # "efficientnet_v2_s",
    # "resnext101_64x4d",
    # "wide_resnet101_2",
    # "regnet_y_1_6gf",
    # "maxvit_t",
    # "swin_v2_s",
    # "shufflenet_v2_x2_0",
    # "vit_l_16"
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
CLASSIFIED_PATH = []
INDEX, CATEGORY, MODEL = [], [], []

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
    vt = VT_[m]

    attacks = [
        MI_FGSM(ALPHA, MI_FGSM_HP[m]["beta"], EPS, adjustment, N_ITER, DEVICE,
#             dim=dim,
#             tim=tim,
#             admix=admix,
#             vt=vt
        ), 
        NI_FGSM(ALPHA, NI_FGSM_HP[m]["beta"], EPS, adjustment, N_ITER, DEVICE,
#             dim=dim,
#             tim=tim,
#             admix=admix,
#             vt=vt
        ),
        AGI_FGSM(ALPHA, AGI_FGSM_HP[m]["delta"], EPS, adjustment, N_ITER, DEVICE,
    #         dim=dim,
    #         tim=tim,
    #         admix=admix,
    #         vt=vt
        ),
        RMSI_FGSM(ALPHA, RMSI_FGSM_HP[m]["beta"], RMSI_FGSM_HP[m]["delta"], EPS, adjustment, N_ITER, DEVICE,
    #         dim=dim,
    #         tim=tim,
    #         admix=admix,
    #         vt=vt
        ),
        AI_FGSM(ALPHA, AI_FGSM_HP[m]["beta_1"], AI_FGSM_HP[m]["beta_2"], AI_FGSM_HP[m]["delta"], EPS, adjustment, N_ITER, DEVICE,
    #         dim=dim,
    #         tim=tim,
    #         admix=admix,
    #         vt=vt
        ),
        NAI_FGSM(ALPHA, NAI_FGSM_HP[m]["beta_1"], NAI_FGSM_HP[m]["beta_2"], NAI_FGSM_HP[m]["delta"], EPS, adjustment, N_ITER, DEVICE,
    #         dim=dim,
    #         tim=tim,
    #         admix=admix,
    #         vt=vt
        )
    ]

    for jdx, att in enumerate(attacks):
        ssim_, psnr_, mse_, nrmse_, acc_, step_ = 0, 0, 0, 0, 0, 0
        classified = []
        os.makedirs(f"{SAVE_DIR}/non_targeted_{m}/{attack_name[jdx]}", exist_ok=True)
        progress = tqdm.tqdm(dataloader, desc=f"{m}_{attack_name[jdx]}")
        for path, image, label, target_label in progress:
            # May cause different result when using mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                adv, predicted, n = att.forward(model, image, label)
#             print(label, predicted)
            acc_ += (label == predicted.cpu()).sum().item()
            
            for kdx in range(image.size(0)):
                if (label[kdx] == predicted.cpu()[kdx]): 
                    classified.append(path[kdx])
                img = (inv_transform(image[kdx]).cpu().numpy().transpose(1, 2, 0))
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                adv_img = (inv_transform(adv[kdx]).cpu().numpy().transpose(1, 2, 0))
                adv_img = np.clip(adv_img, 0, 1)
                matplotlib.image.imsave(f"{SAVE_DIR}/non_targeted_{m}/{attack_name[jdx]}/{os.path.basename(path[kdx])}", adv_img)
                adv_img =  (adv_img * 255).astype(np.uint8)
                ssim = metrics.structural_similarity(img, adv_img, channel_axis=2)
                psnr = metrics.peak_signal_noise_ratio(img, adv_img)
                mse = metrics.mean_squared_error(img, adv_img)
                nrmse = metrics.normalized_root_mse(img, adv_img)

                # print("SSIM : ", ssim)
                # print("PSNR : ",psnr)
                # print("MSE : ", mse)
                # print("NRMSE: ", nrmse)

                ssim_ += ssim
                psnr_ += psnr
                mse_ += mse
                nrmse_ += nrmse
                step_ += n
            progress.set_description(f"{m}_{attack_name[jdx]}/{acc_}")
    

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
        CLASSIFIED_PATH.append('\n'.join(classified))

pd.DataFrame({
    "MODEL":MODEL, "CATEGORY":CATEGORY, "SSIM":SSIM, "PSNR":PSNR, 
    "MSE":MSE, "NRMSE":NRMSE,  "ACC":ACC, "STEP":STEP,
    "CLASSIFIED":CLASSIFIED_PATH
}, index=INDEX).to_csv(f"{SAVE_DIR}/result.csv")

attack_metadata(BATCH_SIZE, ALPHA, EPS, N_ITER, 
                MI_FGSM_HP, NI_FGSM_HP, AGI_FGSM_HP, 
                RMSI_FGSM_HP, AI_FGSM_HP, NAI_FGSM_HP,
                DIM if DIM_ACT else None, 
                TIM if TIM_ACT else None, 
                ADMIX if ADMIX_ACT else None, 
                VT_ if VT_ACT else None)