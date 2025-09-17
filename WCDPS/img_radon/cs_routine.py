import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import odl

def data_consistency_wavelet(
    x1,                      # torch.Tensor [1,1,256,256]，PC步输出的图像域估计
    img,                     # np.ndarray [256,256]，图像域GT（若没有 y_meas 时才用来合成 y）
    model,                   # 'LACT' 或 'SVCT'
    limit_view,              # 例如 90
    niter,                   # PDHG 内迭代步数（建议 5~15）
    lamb=1e-3,               # 小波L1权重（后期可随 sigma(t) 衰减）
    save_root=None,          # 可选：调试可视化目录
    y_meas=None,             # 可选：实测有限角度正弦（优先使用）
    wavelet='db4',           # 小波基
    nlevels=3,               # 小波分解层数
    full_angle=180,          # 全角度（一般 180）
    src_radius=1000, det_radius=100,
):
    """
    目标：min_x 0.5||A x - y||_2^2 + lamb * || W x ||_1   （分析模型）
    解法：PDHG（Chambolle-Pock），K = [A; W]
    返回：torch.Tensor [1,1,256,256]
    """
    device = x1.device if isinstance(x1, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1) ODL 空间与几何 ---
    reco_space = odl.uniform_discr(min_pt=[-50, -50], max_pt=[50, 50],
                                   shape=[256, 256], dtype='float32')
    detector_partition = odl.uniform_partition(-70, 70, 363)

    if model == 'LACT':
        # 前 limit_view 度，连续取 limit_view 个角度
        angle_partition = odl.uniform_partition(0.0, np.deg2rad(limit_view), limit_view)
        angle_partition = odl.uniform_partition(np.deg2rad(180 - limit_view), np.deg2rad(180), limit_view)
    elif model == 'SVCT':
        # 在 0..180 度全范围内稀疏取 limit_view 个角度
        angle_partition = odl.uniform_partition(0.0, np.deg2rad(full_angle), limit_view)
    else:
        raise ValueError("model 必须是 'LACT' 或 'SVCT'")

    geometry  = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                         src_radius=src_radius, det_radius=det_radius)
    A = odl.tomo.RayTransform(reco_space, geometry)

    # --- 2) 构造/获取 y（有限角度正弦） ---
    y = A(img)

    # 可选调试：保存 y / FBP
    if save_root is not None:
        os.makedirs(os.path.join(save_root, 'jpg'), exist_ok=True)
        y_np = np.asarray(y)
        plt.imsave(os.path.join(save_root, 'jpg', 'y_wavelet.png'), y_np, cmap='gray')
        fbp_op   = odl.tomo.fbp_op(A, filter_type='Hann')
        fbpimage = np.asarray(fbp_op(y), dtype=np.float32)
        plt.imsave(os.path.join(save_root, 'jpg', 'fbp_wavelet.png'), fbpimage, cmap='gray')

    # --- 3) 小波变换与算子拼装 ---
    W = odl.trafos.WaveletTransform(reco_space, wavelet=wavelet, nlevels=nlevels, impl='pywt')
    K = odl.BroadcastOperator(A, W)  # K: x -> (A x, W x)

    # f(z1,z2) = 0.5||z1 - y||_2^2 + lamb * ||z2||_1
    data_fit = odl.solvers.L2NormSquared(A.range).translated(y)
    sparsity = lamb * odl.solvers.L1Norm(W.range)
    f = odl.solvers.SeparableSum(data_fit, sparsity)

    # g(x) = 0
    g = odl.solvers.ZeroFunctional(K.domain)

    # --- 4) PDHG 步长（稳定） ---
    op_norm = 1.1 * odl.power_method_opnorm(K)  # 安全冗余
    tau  = 1.0 / op_norm
    sigma= 1.0 / op_norm

    # --- 5) 初始化与求解 ---
    x_np = x1.squeeze().detach().cpu().numpy().astype('float32')  # [256,256]
    x_odl = K.domain.element(x_np)

    # 运行 PDHG
    odl.solvers.pdhg(x_odl, g, f, K, niter=niter, tau=tau, sigma=sigma)

    # --- 6) 回到 Torch ---
    x_np_out = np.asarray(x_odl, dtype=np.float32)
    x_out = torch.from_numpy(x_np_out)[None, None, ...].to(device)
    return x_out


from scipy.fft import fft2, ifft2, fftshift, ifftshift
def frequency_fusion(image1, image2):
    f1 = fft2(image1)
    f2 = fft2(image2)

    # Shift the zero frequency component to the center
    f1_shift = fftshift(f1)
    f2_shift = fftshift(f2)

    # Initialize the fused frequency domain with the first image's frequencies
    fused_freq = np.copy(f1_shift)
    rows, cols = image1.shape
    mask = np.zeros_like(image1)
    mask[:rows // 2, cols // 2:] = 1
    mask[rows // 2:, :cols // 2] = 1
    fused_freq = f1_shift + f2_shift * mask
    fused_freq_shift = ifftshift(fused_freq)
    fused_image = ifft2(fused_freq_shift)

    return np.real(fused_image)