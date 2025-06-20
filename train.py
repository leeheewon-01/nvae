import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
from nvae.vae_celeba import NVAE

class WarmupKLLoss:
    def __init__(self, init_weights, steps, M_N=0.005, eta_M_N=1e-5, M_N_decay_step=3000):
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:
            if self.stage > len(self.steps) - 1:
                break
            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for state AutoEncoder model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each sample batch")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size
    dataset_path = opt.dataset_path

    train_ds = ImageFolderDataset(dataset_path, img_dim=64)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = NVAE(z_dim=512, img_dim=(64, 64))

    # apply Spectral Normalization
    model.apply(add_sn)

    model.to(device)

    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights, map_location=device), strict=False)

    warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                             steps=[4500, 3000, 1500],
                             M_N=opt.batch_size / len(train_ds),
                             eta_M_N=5e-6,
                             M_N_decay_step=36000)
    print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    step = 0
    total_loss = 0
    # Epoch progress bar

    for epoch in range(epochs):
        model.train()

        batch_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}')
        for i, image in enumerate(batch_pbar):

            optimizer.zero_grad()

            image = image.to(device)
            image_recon, recon_loss, kl_losses = model(image)
            kl_loss = warmup_kl.get_loss(step, kl_losses)
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            batch_pbar.set_description(f'Epoch {epoch}/{epochs} - Step {step} - Loss {loss.item():.6f}')
            batch_pbar.update(1)

            step += 1
            total_loss += loss.item()

            # if step != 0 and step % 100 == 0:
            #     with torch.no_grad():
            #         z = torch.randn((1, 512, 2, 2)).to(device)
            #         gen_img, _ = model.decoder(z)
            #         gen_img = gen_img.permute(0, 2, 3, 1)
            #         gen_img = gen_img[0].cpu().numpy() * 255
            #         gen_img = gen_img.astype(np.uint8)

            #         plt.imshow(gen_img)
            #         # plt.savefig(f"output/ae_ckpt_%d_%d.png" % (epoch, step))
            #         plt.show()

        scheduler.step()

        torch.save(model.state_dict(), f"checkpoints/ae_ckpt_%d_%.6f.pth" % (epoch, total_loss))

        total_loss = 0
        # model.eval()

        # with torch.no_grad():
        #     z = torch.randn((1, 512, 2, 2)).to(device)
        #     gen_img, _ = model.decoder(z)
        #     gen_img = gen_img.permute(0, 2, 3, 1)
        #     gen_img = gen_img[0].cpu().numpy() * 255
        #     gen_img = gen_img.astype(np.uint8)

        #     plt.imshow(gen_img)
        #     plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, loss.item()))
        #     plt.show()
