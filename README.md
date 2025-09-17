# wcdps

**Wavelet Consistent Diffusion Posterior Sampling (WCDPS) for LACT/SVCT**  
*Tweedie anchor refinement with PDHG wavelet sparsity, sketch-based init, and CG data consistency.*

## Quickstart

```bash
git clone https://github.com/wenwen2001/wcdps.git
cd wcdps
python WCDPS/0_sino_recon_img.py
`\`\`\``
## Directory Structure
WCDPS/
├─ 0_sino_recon_img.py            # Example entry script
└─ img_radon/
   ├─ configs/                    # Experiment/config files
   ├─ models/                     # Model definitions (ddpm, ncsnpp, unet, ema, etc.)
   ├─ physics/                    # Operators / Radon transforms / filters
   ├─ op/                         # Operator wrappers
   ├─ sampling.py
   ├─ sde_lib.py
   ├─ utils.py
   └─ wcdps_utils.py
