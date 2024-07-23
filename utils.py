import tqdm
import torch
import json, os
from piq.ssim import ssim 
from piq.fsim import fsim
from piq.psnr import psnr
from piq.vif import vif_p
from piq.ms_ssim import multi_scale_ssim
from piq.iw_ssim import information_weighted_ssim
from piq.mdsi import mdsi
from prettytable import PrettyTable

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

def attack_metadata(save_dir, time, amp, batch_size, alpha, eps, n_iter, act, num_fig,
                    performed_attck, targeted,
                    mi_fgsm=None, ni_fgsm=None, agi_fgsm=None, adi_fgsm=None,
                    rmsi_fgsm=None, ai_fgsm=None, ani_fgsm=None, nai_fgsm=None,
                    yogi_fgsm=None, adai_fgsm=None, abi_fgsm=None,
                    dim=None, tim=None, admix=None, vt=None):
    metadata = {
        "Time":time,
        "Enable_amp":amp,
        "Hyperparameters":{"batch_size":batch_size, "alpha":alpha, "eps":eps, "n_iter":n_iter,
                        "activation":act, "num_figure":num_fig},
        "Attack":performed_attck,
        "Targeted":targeted,
        "DIM":dim,
        "TIM":tim,
        "ADMIX":admix,
        "VT":vt,
        "MI-FGSM":mi_fgsm,
        "NI-FGSN":ni_fgsm,
        "AGI-FGSM":agi_fgsm,
        "ADI-FGSM":adi_fgsm,
        "RMSI-FGSM":rmsi_fgsm,
        "AI-FGSM":ai_fgsm,
        "ANI-FGSM":ani_fgsm,
        "NAI-FGSM":nai_fgsm,
        "YOGI":yogi_fgsm, 
        "ADAI":adai_fgsm,
        "ADABELIEF":abi_fgsm,
    }
    time = time.replace("/", "_")
    time = time.replace(":", "_")
    with open(f'{save_dir}/metadata.json', 'w') as file:
        json.dump(metadata, file, indent=4)
    print(metadata["Hyperparameters"], metadata["Targeted"])
    return json.dumps(metadata, indent=4)

def compute_metrics(img, adv_img, verbose=False):
    '''
    return: ssim, psnr, fsim, iw-ssim, ms-ssim, mdsi, vifp, mse, mae
    '''
    _ssim = ssim(img, adv_img, data_range=1.0).item()
    _psnr = psnr(img, adv_img, data_range=1.0).item()
    _fsim = fsim(img, adv_img, data_range=1.0).item()
    _iw_ssim = information_weighted_ssim(img, adv_img, data_range=1.0).item()
    _ms_ssim = multi_scale_ssim(img, adv_img, data_range=1.0).item()
    _mdsi = mdsi(img, adv_img, data_range=1.0).item()
    _vifp = vif_p(img, adv_img, data_range=1.0).item()
    _mse = float(torch.mean((img - adv_img) ** 2).cpu())
    _mae = float(torch.mean(torch.abs(img - adv_img)).cpu())
    if verbose: 
        print(f'SSIM: {_ssim}')
        print(f'PSNR: {_psnr}')
        print(f'FSIM: {_fsim}')
        print(f'IW-SSIM: {_iw_ssim}')
        print(f'MS-SSIM: {_ms_ssim}')
        print(f'MDSI: {_mdsi}')
        print(f'VIFp: {_vifp}')
        print(f'MSE: {_mse}')
        print(f'MAE: {_mae}')
    return [_ssim, _psnr, _fsim, _iw_ssim, _ms_ssim, _mdsi, _vifp, _mse, _mae]

def display_result(m, att, targeted, metrics, metrics_average):
    t = PrettyTable(["ITEM", "VALUE"])
    t.add_row(['METHOD', f"{m}_{att}"])
    t.add_row(["TARGETED", targeted])
    for idx in range(len(metrics)):
        t.add_row([metrics[idx], metrics_average[idx]])
    print(t)      

def read_metadata(dir):
    files = os.listdir(dir)    
    json_file = None
    for file in files:
        if file.endswith('.json'):
            json_file = file
            break    
    if json_file is None:
        raise FileNotFoundError("No metadata JSON file found in the directory")    
    json_path = os.path.join(dir, json_file)
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata


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

