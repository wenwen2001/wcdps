import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import torch
import numpy as np
from img_radon.sampling import get_corrector, get_predictor
from img_radon.physics.ct import CT
import wcdps_utils
import img_radon.recon_solver as recon_solver
from pathlib import Path
from img_radon.configs.ve import AAPM_256_ncsnpp as configs  # Load config
from sde_lib import VESDE
from img_radon.models import utils as mutils

def run_wcdps(
    gt_image: np.ndarray,
    alpha,
    save_root=None,
    ckpt_root_path: Path = Path("img_radon/checkpoint/score-sde/CT"),
    ckpt_name: str = "checkpoint_199.pth",
    method: str = "wcdps",
    recon_size: int = 256,
    degree: int = 90,
    reverse_t: int = 2000,
    sino_noise: float = 0.0,
    device: str = "cuda:0",
    init_image: torch.Tensor = None,
    ID:str = 'L067_0.npy',
    degree_model:str ='LACT',

):

    config_name = "AAPM_256_ncsnpp"
    config = configs.get_config()
    config.device = torch.device(device)

    # Define view angles
    # 定义LACT角度集 "LACT"
    if degree_model=='LACT':
        view_full_num = 180
        view_limited_num = int(view_full_num * (degree / 180))
        view_limited_list = np.linspace(0, view_limited_num, view_limited_num, endpoint=False, dtype=int)
    elif degree_model=='SVCT':
        view_full_num = 180
        view_limited_num = int(view_full_num * (degree / 180))
        view_limited_list = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)
    # Define radon operators
    radon_torch_mod = CT(
        img_width=recon_size,
        used_view_list=view_limited_list,
        view_full_num=view_full_num,
        device=config.device,
    )
    radon_torch_mod_all = CT(
        img_width=recon_size,
        used_view_list=np.linspace(0, 180, 180, endpoint=False, dtype=int),
        view_full_num=180,
        device=config.device,
    )

    gt_image_np = gt_image
    gt_image = torch.from_numpy(gt_image).float().to(config.device).view(1, 1, recon_size, recon_size)
    measurement = radon_torch_mod.A_LV(gt_image).float().to(config.device).clone().detach()
    fbp_lv = radon_torch_mod.FBP_LV(measurement)

    # Load model
    #
    ckpt_filename = ckpt_root_path / ckpt_name
    sigmas = mutils.get_sigmas(config)
    sde = VESDE(
        sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales
    )
    score_model = wcdps_utils.get_models(config, str(ckpt_filename))

    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)

    # Run different reconstruction pipelines
    if method == "wcdps":
        recon_pipeline = recon_solver.get_image_domain_wcdps(
            sde, predictor, corrector, config=config, radon=radon_torch_mod,
            eps=1e-10, rho=5, lamb=0.2, niter_CG=1, niter_ADMM=1, data_lamb=1)
        x,dc_init_img,psnr, ssim = recon_pipeline(model=score_model,
                           measurement=measurement,
                           init_image=fbp_lv,  # fbp_lv, init_image
                           reverse_t=reverse_t,  # args.reverse_t, ##2000
                           save_path=save_root,
                           final_consistency=False,
                           gt=gt_image_np,
                           sino_reconfbp=fbp_lv,
                           degree=degree,
                            degree_model=degree_model,
                           alpha=alpha,ID=ID,
                            Tweedie = False,
                           )
    elif method == "Wavelet_PDHG":
        recon_pipeline = recon_solver.get_Wavelet_dataconsistency(
            radon=radon_torch_mod)
        x,dc_init_img, psnr, ssim = recon_pipeline(
                           save_path=save_root,
                           gt=gt_image_np,
                           sino_reconfbp=fbp_lv,
                           degree=degree,
                            degree_model=degree_model,
                           ID=ID,
                            niter=100,
                           )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return x,dc_init_img,psnr, ssim


