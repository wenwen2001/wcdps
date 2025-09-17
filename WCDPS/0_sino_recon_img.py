from img_radon.single_image_domain_run_WCDPS import run_wcdps
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import csv
import random

# settings
limit_angle =10
angle_model='SVCT'
reverse_t_list = [1000] #T'
method_list= ['WCDPS']# method:   Wavelet_PDHG 'DPS'  PSDM  MCG
num_samples=100
dataset='kits23'#'kits23' 'covid'

# 定义保存结果的txt文件路径
log_file=os.path.join('./results-imagedomain','TXT',f'psnr_ssim_log_{limit_angle}_{angle_model}.txt')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"PSNR && SSIM{angle_model} {limit_angle}\n")
    f.write("====================\n")
#######################################################################################
#配置参数
## 图像域路径


## 设置数据集
if dataset == 'AAPM':
    dataset_path = 'test_data/L333_608.npy'
elif dataset == 'kits23':
    dataset_path = ''

## 设置采样数据编号

file_list = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
Name =[f.split('_')[0].replace('.npy', '') for f in file_list]
id_list = [f.split('_')[1].replace('.npy', '') for f in file_list]
random_indices = random.sample(range(len(file_list)), num_samples)
selected_ids = [id_list[i] for i in random_indices]


#######################################################################################
# #开始训练
ssim_results = []
psnr_results = []
for index,NO in enumerate(selected_ids):
    ID = f'{Name}_{NO}.npy'
    name=f'{Name}_{NO}'

    gt_img_path = os.path.join(dataset_path, ID)
    gt_image = np.load(gt_img_path) #图像域重建用到

    for method in method_list:
        for reverse_t in reverse_t_list:
            print(f"处理进度{index}/{len(selected_ids)}; 测试任务：{method}_{angle_model}:{limit_angle}_{reverse_t}")
            recon_img,dc_init_img,psnr, ssim = run_wcdps(
                ID = ID,
                gt_image=gt_image,
                alpha=99,
                save_root=r"./results-imagedomain",
                ckpt_root_path= Path("img_radon/checkpoint/score-sde/CT"),
                ckpt_name = "checkpoint_299.pth",
                method=method,
                recon_size=256,
                degree=limit_angle,
                degree_model=angle_model,#'SVCT'
                init_image=None,#sino_radon_recon_fbp
                reverse_t= reverse_t,
            ) #返回tensor
            print(f"{name}: PSNR:{psnr}; SSIM: {ssim:.6f}")
            recon_img_np=recon_img.detach().squeeze().cpu().numpy()
            plt.imsave(os.path.join('./results-imagedomain',f'result_jpg_{limit_angle}_{angle_model}',f"{name}_recon.jpg"),recon_img_np, cmap=plt.cm.gray)

            psnr_results.append(psnr)
            ssim_results.append(ssim)
            log_line = f"{index}/{len(selected_ids)}:{name}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f};;(t={reverse_t})\n"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)

            print("--------------------------------------------------------------------------------")

        #######################################################################################

