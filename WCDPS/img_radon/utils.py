from pathlib import Path
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev

def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return np.transpose(x, (1, 2, 0))


def clear(x, normalize=True):
    x = x.detach().cpu().squeeze().numpy()
    if normalize:
        x = normalize_np(x)
    return x


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
    ckpt_dir = Path(ckpt_dir)
    # import ipdb; ipdb.set_trace()
    # ckpt = ckpt_dir / "checkpoint.pth"
    if not ckpt_dir.exists():
        logging.warning(f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if not skip_optimizer:
            state["optimizer"].load_state_dict(loaded_state["optimizer"])
        loaded_model_state = loaded_state["model"]
        if skip_sigma:
            loaded_model_state.pop("module.sigmas")

        state["model"].load_state_dict(loaded_model_state, strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        print(f"    loaded checkpoint dir from {ckpt_dir}")
        return state


def restore_checkpoint_clx(ckpt_dir, ckpt_dir2, state, device, skip_sigma=False, skip_optimizer=False):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir2 = Path(ckpt_dir2)
    # import ipdb; ipdb.set_trace()
    # ckpt = ckpt_dir / "checkpoint.pth"
    if not ckpt_dir.exists():
        logging.warning(f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        print(len(loaded_state))
        loaded_state2 = torch.load(ckpt_dir2, map_location=device)
        state["model"].load_state_dict(dict(loaded_state[0]), strict=False)

        # state["model"].load_state_dict(loaded_state[0], strict=True)
        # state['model'].load_state_dict(loaded_model_state, strict=False)
        state["ema"].load_state_dict(loaded_state[-1])
        state["step"] = loaded_state2["step"]
        print(f"loaded checkpoint dir from {ckpt_dir}")
        return state


def save_checkpoint(ckpt_dir, state, name="checkpoint.pth"):
    ckpt_dir = Path(ckpt_dir)
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir / name)


"""
Helper functions for new types of inverse problems
"""


def crop_center(img, cropx, cropy):
    c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def normalize(img):
    """Normalize img in arbitrary range to [0, x]"""
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def normalize_np(img):
    """Normalize img in arbitrary range to [0, x]"""
    img -= np.min(img)
    img /= np.max(img)
    return img


def normalize_np_kwarg(img, maxv=1.0, minv=0.0):
    """Normalize img in arbitrary range to [0, x]"""
    img -= minv
    img /= maxv
    return img


def normalize_complex(img):
    """normalizes the magnitude of complex-valued image to range [0, x]"""
    abs_img = normalize(torch.abs(img))
    # ang_img = torch.angle(img)
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def batchfy(tensor, batch_size):
    n = len(tensor)
    num_batches = n // batch_size + 1
    return tensor.chunk(num_batches, dim=0)


def img_wise_min_max(img):
    img_flatten = img.view(img.shape[0], -1)
    img_min = torch.min(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)
    img_max = torch.max(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)

    return (img - img_min) / (img_max - img_min)


def patient_wise_min_max(img):
    std_upper = 3
    img_flatten = img.view(img.shape[0], -1)

    std = torch.std(img)
    mean = torch.mean(img)

    img_min = torch.min(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)
    img_max = torch.max(img_flatten, dim=-1)[0].view(-1, 1, 1, 1)

    min_max_scaled = (img - img_min) / (img_max - img_min)
    min_max_scaled_std = (std - img_min) / (img_max - img_min)
    min_max_scaled_mean = (mean - img_min) / (img_max - img_min)

    min_max_scaled[min_max_scaled > min_max_scaled_mean + std_upper * min_max_scaled_std] = 1

    return min_max_scaled


def create_sphere(cx, cy, cz, r, resolution=256):
    """
    create sphere with center (cx, cy, cz) and radius r
    """
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x, y, z])


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


def image_grid_gray(x, size=32):
    img = x.reshape(-1, size, size)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size)).transpose((0, 2, 1, 3)).reshape((w * size, w * size))
    return img


def show_samples_gray(x, size=32, save=False, save_fname=None):
    x = x.detach().cpu().numpy()
    img = image_grid_gray(x, size=size)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(img, cmap="gray")
    plt.show()
    if save:
        plt.imsave(save_fname, img, cmap="gray")


def get_mask(img, size, batch_size, type="gaussian2d", acc_factor=8, center_fraction=0.04, fix=False):
    mux_in = size**2
    if type.endswith("2d"):
        Nsamp = mux_in // acc_factor
    elif type.endswith("1d"):
        Nsamp = size // acc_factor
    if type == "gaussian2d":
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == "uniformrandom2d":
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
                # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == "gaussian1d":
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from : c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from : c_from + Nsamp_center] = 1
    elif type == "uniform1d":
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from : c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from : c_from + Nsamp_center] = 1
    else:
        NotImplementedError(f"Mask type {type} is currently not supported.")

    return mask



def save_data(fname, arr):
    """Save data as .npy and .png"""
    np.save(fname + ".npy", arr)
    plt.imsave(fname + ".png", arr, cmap="gray")


def mean_std(vals: list):
    return mean(vals), stdev(vals)
