import torch
import torch.nn.functional as F
from pathlib import Path
import odl
from matplotlib import pyplot as plt
from img_radon.models import utils as mutils
from img_radon.models.ema import ExponentialMovingAverage
from img_radon.utils import restore_checkpoint
from torch.utils.data import Dataset, DataLoader
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', str(s))]

class AAPM(Dataset):
  def __init__(self, root, sort):
    self.root = root
    # self.data_list = list(root.glob('*.npy'))
    self.data_list = list(root.glob('L506_*.npy'))
    self.sort = sort
    if sort:
      self.data_list = sorted(self.data_list, key=natural_sort_key)  # 用自然排序！

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname)
    data = np.expand_dims(data, axis=0)
    return data, fname.name

def sampling_create_dataloader(configs,shuffle=False, sort=True,batch_size=10):
    train_dataset = AAPM(Path(configs.data.root) / f'np_256_img_dataset', sort=sort) #data.root = '/media/harry/tomo/AAPM_data/256/train'
    val_dataset = AAPM(Path(configs.data.root) / f'np_256_img_dataset_test', sort=sort) #data.root = '/media/harry/tomo/AAPM_data/256/test'

    train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=True
    )
    val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=True
    )
    return train_loader, val_loader


def psnr_ssim(gt_arr, recon_arr):
    """
     PSNR  SSIM
   
        gt_arr: numpy [B, 1, H, W]、[1, H, W] or [H, W]
        recon_arr:  gt_arr
    return：
        psnr_list, ssim_list：
    """
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


from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim,mean_squared_error as mse

def indicate(img1,img2):
    if len(img1.shape) == 3:
        batch = img1.shape[0]
        psnr0 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)
        for i in range(batch):
            t1= img1[i,...]/np.max(img1[i,...])
            t2= img2[i,...]/np.max(img2[i,...])
            psnr0[i,...] = psnr(t1,t2,data_range=1)
            ssim0[i,...] = ssim(t1,t2,data_range=1.0)
            mse0[i,...] = mse(t1,t2)
        return psnr0,ssim0,mse0
    else:
        # print("len(img1.shape) == 2")
        img1 /= img1.max()
        img2 /= img2.max()
        psnr0 = psnr(img1,img2,data_range=1)
        ssim0 = ssim(img1,img2,data_range=1.0)
        mse0 = mse(img1,img2)
        return psnr0,ssim0,mse0

def img_normalized_tensor(img: torch.Tensor):
  B = img.shape[0]
  out = torch.empty_like(img)
  for i in range(B):
    x = img[i]
    min_val = x.min()
    max_val = x.max()
    out[i] = (x - min_val) / (max_val - min_val + 1e-8)
  return out

class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass


class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb

def data_consistency(x1,img,model,limit_view,niter,lamb=0.0001,save_root=None):
    """
    img--np
    lamb 表示TV正则化所占权重
    """

    # print("model==la_sino")

    reco_space = odl.uniform_discr(
                min_pt=[-50, -50], max_pt=[50, 50], shape=[256, 256], dtype='float32')
    detector_partition = odl.uniform_partition(-70, 70, 363)
    if model=='LACT':
        angle_partition = odl.uniform_partition(0 , limit_view * np.pi / 180, limit_view)
        angle_partition = odl.uniform_partition(np.deg2rad(180-limit_view), np.deg2rad(180), limit_view)
    elif model=='SVCT':
        angle_partition = odl.uniform_partition(0, np.deg2rad(180), limit_view)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=1000, det_radius=100)
    # Create the forward operator A
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

    # Save FBP img
    fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')
    data = ray_trafo(img)   # y=Ax data是[90,363]的正弦图像
    # print(f"data shape: {data.shape}");print(data.dtype)
    plt.imsave(os.path.join(save_root,'jpg','y.png'), data, cmap='gray')
    fbpimage = fbp_op(data)  #
    fbpimage = np.array(fbpimage, dtype=np.float32)
    plt.imsave(os.path.join(save_root,'jpg','odl_fbpimage.png'), fbpimage, cmap='gray')
    # import pdb; pdb.set_trace()

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
    # Choose a starting point 
    # x = L.domain.zero()
    # plt.imsave('./results/0-x.png', x.asarray(), cmap='gray')
    # Run the algorithm
    # odl.solvers.admm_linearized(x, f, g, L, tau, sigma, niter, callback=callback)
    # odl.solvers.admm_linearized(reco_space.element(x1.squeeze().cpu().numpy()), f, g, L, tau, sigma, niter
    #                             )
    x1 = x1.squeeze().cpu().numpy()
    x1 = L.domain.element(x1)
    x = x1
    odl.solvers.admm_linearized(x, f, g, L, tau=tau, sigma=sigma, niter=niter                                    )
    x = np.array(x, dtype=np.float32)
    x = x.copy()
    x = torch.from_numpy(x)
    x = x.unsqueeze(0).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # x_mean = x
    # # Display images
    # data_np = data.asarray()
    # x_np = x.asarray()
    # plt.imsave('./results/11-gt.png', gt, cmap='gray')
    # plt.imsave('./results/22-data_np.png', data_np, cmap='gray')
    # plt.imsave('./results/33-x_np.png', x_np, cmap='gray')



    return x


def get_models(config, ckpt_path):
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_path, state, config.device,skip_sigma=False,  skip_optimizer=True)
    ema.copy_to(score_model.parameters())

    return score_model

