import os
import urllib.request
from pathlib import Path
from PIL import Image

from astropy.io import fits
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import pytorch_lightning as pl
from torch import optim, utils
from pytorch_lightning.callbacks import ModelCheckpoint

galaxy_images_cropped_downsampled = np.load("../data/galaxy_images_cropped_downsampled.npy")
# galaxy_images_cropped_downsampled = np.interp(galaxy_images_cropped_downsampled, (galaxy_images_cropped_downsampled.min(), galaxy_images_cropped_downsampled.max()), (-1, +1))

model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    channels = 1,
    timesteps = 200,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

class LitDiffusion(pl.LightningModule):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion

    def training_step(self, batch, batch_idx):
        loss = self.diffusion(batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
LGD = LitDiffusion(diffusion)
from pathlib import Path
ckpt = max(list(Path('lightning_logs').rglob("*.ckpt")), key=os.path.getctime)
print(ckpt)
LGD = LitDiffusion.load_from_checkpoint(ckpt, diffusion=diffusion)

dataset = torch.tensor(galaxy_images_cropped_downsampled).unsqueeze(1).float()
train_loader = utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

epochs = 0
checkpoint_callback = ModelCheckpoint(monitor="train_loss")
trainer = pl.Trainer(gpus=-1, max_epochs=epochs, strategy='ddp_find_unused_parameters_false', callbacks=[checkpoint_callback], limit_train_batches=50)
trainer.fit(model=LGD, train_dataloaders=train_loader)



diffusion = LGD.diffusion.cuda()
diffusion.eval()

sampled_images = diffusion.sample(batch_size = 8).squeeze().cpu().detach().numpy()
real_images = galaxy_images_cropped_downsampled[np.random.randint(len(galaxy_images_cropped_downsampled), size=8)]
images = np.concatenate([sampled_images, real_images], axis=0)

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(9., 9.))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4),axes_pad=0.1,)
for ax, im in zip(grid, images):
    ax.imshow(im, cmap='magma')
plt.savefig('results.png')