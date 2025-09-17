import functools
import os
import odl
from utils_origin import psnr_ssim
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from img_radon.physics.ct import *
from img_radon.sampling import shared_corrector_update_fn, shared_predictor_update_fn
from img_radon.wcdps_utils import img_normalized_tensor
from wcdps_utils import data_consistency
from img_radon.cs_routine import data_consistency_wavelet

def get_image_domain_wcdps(sde, predictor, corrector, config, radon,
                           rho, lamb, data_lamb, niter_ADMM, niter_CG, eps):
    def _A(x):
        return radon.A_LV(x)

    def _AT(sinogram):
        return radon.BP_LV(sinogram)


    def CG(A_fn, b_cg, x, n_inner=niter_CG):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def A_cg(x):
        return _AT(_A(x))

    def CS_routine(x, ATy, niter=niter_ADMM, n_inner=niter_CG):
        for i in range(niter):
            b_cg = ATy
            x = CG(A_cg, b_cg, x, n_inner=n_inner)
        return x
#############################################
# Define predictor & corrector
################################################

    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=config.training.continuous,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
    )

    eps = 1e-10

    def get_unconditon_update_fn(update_fn):
        def unconditon_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_mean_next, score = update_fn(x, vec_t, model=model)

            return x_next, x_mean_next, score

        return unconditon_update_fn

    predictor_denoise_update_fn = get_unconditon_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_unconditon_update_fn(corrector_update_fn)

    def wcdps_recon(model, init_image, measurement, ID, reverse_t, gt, sino_reconfbp,
                    degree, save_path, alpha, degree_model, final_consistency=False, data_lamb=data_lamb,
                    niter_CG=niter_CG, niter_ADMM=niter_ADMM,
                    Tweedie=True,
                    alpha_start_frac=0.1,
                    TR=50,

                    ):

        init_image = data_consistency(x1=img_normalized_tensor(sino_reconfbp), img=gt, model=degree_model, limit_view=degree, niter=100,save_root=save_path)
        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N) #T=x,eps1e-10，N=2000
            # =====  SI strategy =====
            if reverse_t == sde.N:
                x = sde.prior_sampling(init_image.shape).to(measurement.device) #x是256大小的随机噪声
                start_idx = 0
            else:
                start_idx = sde.N - reverse_t
                vec_reverse_t = timesteps[-reverse_t].to(measurement.device)
                x = sde.prior_sampling_t(init_image, vec_reverse_t).to(measurement.device)
            plt.imsave(os.path.join(save_path, 'jpg',f'x_00.jpg'), x.squeeze().cpu().numpy(),
                       cmap='gray')
            trigger_from = start_idx + int((sde.N - start_idx) * alpha_start_frac)
            for i in tqdm(range(start_idx, sde.N), colour="blue", unit="step", smoothing=0): #sde.N
                t = timesteps[i]
                # =====  PC sampling =====
                x, _, _ = predictor_denoise_update_fn(model, x, t)
                x, x_mean, score_c = corrector_denoise_update_fn(model, x, t)
                trigger = (i >= trigger_from) and ((i - start_idx) % TR == 0) and (i > start_idx)
                # =====  AR strategy =====
                if trigger:
                    sigma_tweedie = sde.sigma_t(t)
                    x = x_mean + (sigma_tweedie ** 2) * score_c
                    x = data_consistency_wavelet(x1=x, img=gt, model=degree_model, limit_view=degree, niter=25, save_root=save_path)
                # =====  DC =====
                with torch.enable_grad():
                    ATy = _AT(measurement)
                    x = CS_routine(x, ATy, niter=niter_ADMM,n_inner=niter_CG)
                plt.imsave(os.path.join(save_path, 'progress', f'x{i}.jpg'), x.squeeze().cpu().numpy(),cmap='gray')


            sigma_tweedie = sde.sigma_t(t)
            x = x_mean + (sigma_tweedie ** 2) * score_c
            x = data_consistency(x1=x, img=gt, model=degree_model,limit_view=degree, niter=100, save_root=save_path)
            psnr, ssim = psnr_ssim(x.squeeze().cpu().numpy(), gt)
            if final_consistency:
                x = data_consistency_wavelet(x1=x, img=gt, model=degree_model, limit_view=degree, niter=25,save_root=save_path)
                psnr, ssim = psnr_ssim(x.squeeze().cpu().numpy(), gt)
            if Tweedie:
                sigma_tweedie=sde.sigma_t(t)
                x = x + (sigma_tweedie ** 2) * score_c
            return x,init_image,psnr, ssim

    return wcdps_recon


def get_Wavelet_dataconsistency(radon):
    def recon(sino_reconfbp,gt,degree_model,degree,niter,save_path,ID):
        x = data_consistency_wavelet(x1=img_normalized_tensor(sino_reconfbp), img=gt, model=degree_model,
                                              limit_view=degree, niter=niter, save_root=save_path)
        psnr, ssim = psnr_ssim(x.squeeze().cpu().numpy(), gt)
        return x,sino_reconfbp, psnr, ssim

    return recon
