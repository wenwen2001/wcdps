import torch
import numpy as np
import random
from scipy.ndimage import uniform_filter
def img_normalized_tensor(img: torch.Tensor):
  """
  对 [B, C, H, W] 的张量按样本逐个归一化到 [0, 1]
  """
  B = img.shape[0]
  out = torch.empty_like(img)
  for i in range(B):
    x = img[i]
    min_val = x.min()
    max_val = x.max()
    out[i] = (x - min_val) / (max_val - min_val + 1e-8)  # 避免除以0
  return out
def calculate_matrix(results_all):
    psnr_ssim_values = [psnr_ssim for _, _, psnr_ssim in results_all]
    psnr_ssim_values = np.array(psnr_ssim_values)
    # 计算指标
    mean = np.mean(psnr_ssim_values)
    max = np.max(psnr_ssim_values)
    min = np.min(psnr_ssim_values)
    return mean, max, min
def psnr_ssim(gt_arr, recon_arr):
    def compute_single_psnr_ssim(gt, recon):
        mse = np.mean((gt - recon) ** 2)
        max_pixel_value = max(np.max(gt), np.max(recon))
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse + 1e-8)

        gt = gt.astype(np.float64)
        recon = recon.astype(np.float64)
        L = max_pixel_value
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        mu1 = uniform_filter(gt, size=11)
        mu2 = uniform_filter(recon, size=11)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = uniform_filter(gt * gt, size=11) - mu1_sq
        sigma2_sq = uniform_filter(recon * recon, size=11) - mu2_sq
        sigma12 = uniform_filter(gt * recon, size=11) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = np.mean(ssim_map)
        return psnr, ssim

    # 自动去掉 batch 和 channel 的维度
    gt_arr = np.squeeze(gt_arr)
    recon_arr = np.squeeze(recon_arr)

    if gt_arr.ndim == 2:
        return compute_single_psnr_ssim(gt_arr, recon_arr)
    elif gt_arr.ndim == 3:  # [B, H, W]
        psnr_list, ssim_list = [], []
        for i in range(gt_arr.shape[0]):
            psnr, ssim = compute_single_psnr_ssim(gt_arr[i], recon_arr[i])
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        return psnr_list, ssim_list
    else:
        raise ValueError(f"输入维度不支持: {gt_arr.shape}")
def img_normalized(img):
  return (img - np.min(img)) / (np.max(img) - np.min(img))
def data_consistency(x1,img,model,limit_view,niter,lamb=0.0001):
    """
  
    """
    if model=='LACT':
        reco_space = odl.uniform_discr(
                    min_pt=[-50, -50], max_pt=[50, 50], shape=[256, 256], dtype='float32')
        detector_partition = odl.uniform_partition(-70, 70, 363)
        angle_partition = odl.uniform_partition(0 , limit_view * np.pi / 180, limit_view)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=1000, det_radius=100)
        # Create the forward operator A
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

        # Save FBP img
        fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')
        data = ray_trafo(img)   # y=Ax data是[90,363]的正弦图像
        # print(f"data shape: {data.shape}")
        fbpimage = fbp_op(data)  #
        fbpimage = np.array(fbpimage, dtype=np.float32)
        # plt.imsave('./results/00-fbpimage.png', fbpimage, cmap='gray')

        grad = odl.Gradient(reco_space)
        L = odl.BroadcastOperator(ray_trafo, grad)
        f = odl.solvers.ZeroFunctional(L.domain)

        data_fit = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
        reg_func = lamb * odl.solvers.L1Norm(grad.range)
        g = odl.solvers.SeparableSum(data_fit, reg_func)

        op_norm = 1.1 * odl.power_method_opnorm(L) #, maxiter=20
        sigma = 2.0  # Step size for g.proximal
        tau = sigma / op_norm ** 2  # Step size for f.proximal
        # tau = 1 / op_norm  # Step size for the primal variable 1.0
        # sigma = 1 / op_norm  # Step size for the dual variable 1.0

        callback = (odl.solvers.CallbackPrintIteration(step=10) &
                    odl.solvers.CallbackShow(step=10))

        x1 = x1.squeeze().cpu().numpy()
        x1 = L.domain.element(x1)
        x = x1
        odl.solvers.admm_linearized(x, f, g, L, tau=tau, sigma=sigma, niter=niter                                    )
        x = np.array(x, dtype=np.float32)
        x = x.copy()
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return x

    elif model=='SVCT':
        # —— 1) 定义重建空间与几何（稀疏视角：0..π 上均匀抽样 limit_view 个角度）——
        reco_space = odl.uniform_discr(
            min_pt=[-50, -50], max_pt=[50, 50], shape=[256, 256], dtype='float32')

        detector_partition = odl.uniform_partition(-70, 70, 363)
        # 稀疏视角（Sparse-View）：角度覆盖 0..π，但仅有 limit_view 个等间隔角度
        angle_partition = odl.uniform_partition(0, np.pi, limit_view)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=1000, det_radius=100)

        # —— 2) 前向与 FBP（用于生成数据项与可视化）——
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
        fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')

        # img 是 numpy 的 2D 图像（GT 或上一阶段重建），确保类型/域一致
        img_odl = reco_space.element(np.array(img, dtype=np.float32))
        data = ray_trafo(img_odl)  # 稀疏视角下的投影数据 y = A x

        fbpimage = fbp_op(data)  # 稀疏视角的 FBP 结果（可作为参考）
        fbpimage = np.array(fbpimage, dtype=np.float32)
        # plt.imsave('./results/00-fbpimage_svct.png', fbpimage, cmap='gray')

        # —— 3) 构造 ADMM-Linearized 问题：min_x  0.5||Ax - y||_2^2 + λ ||∇x||_1 ——
        grad = odl.Gradient(reco_space)
        L = odl.BroadcastOperator(ray_trafo, grad)

        f = odl.solvers.ZeroFunctional(L.domain)

        data_fit = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
        reg_func = lamb * odl.solvers.L1Norm(grad.range)
        g = odl.solvers.SeparableSum(data_fit, reg_func)

        # 步长参数（与 LACT 分支保持一致的风格）
        op_norm = 1.1 * odl.power_method_opnorm(L)
        sigma = 2.0
        tau = sigma / op_norm ** 2

        # —— 4) 以 x1 为初值做校正 ——
        x_np = x1.squeeze().cpu().numpy().astype(np.float32)
        x = reco_space.element(x_np)

        odl.solvers.admm_linearized(x, f, g, L, tau=tau, sigma=sigma, niter=niter)

        # 回到 torch 张量（CUDA）
        x = np.array(x, dtype=np.float32)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return x
