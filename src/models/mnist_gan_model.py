from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

        # Lightning 2.0 requires manual optimization when using multiple optimizers
        self.automatic_optimization = False

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx) -> None:
        opt_g, opt_d = self.optimizers()

        log_dict, g_loss, d_loss = self.step(batch, batch_idx)

        # Train Generator
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Train Discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict(
            {"/".join(("train", k)): v for k, v in log_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].shape[0],
        )

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, _, _ = self.step(batch, batch_idx)
        self.log_dict(
            {"/".join(("val", k)): v for k, v in log_dict.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch[0].shape[0],
        )

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, _, _ = self.step(batch, batch_idx)
        self.log_dict(
            {"/".join(("test", k)): v for k, v in log_dict.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch[0].shape[0],
        )

    def step(self, batch, batch_idx) -> Tuple[Dict[str, Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Single step that computes losses for both generator and discriminator.
        Returns log_dict, g_loss, d_loss — callers use what they need
        (training uses both losses for backward; val/test only need log_dict).
        """

        imgs, labels = batch
        batch_size = imgs.shape[0]

        # adversarial ground truths
        valid = torch.ones((batch_size, 1), device=self.device, dtype=imgs.dtype)
        fake = torch.zeros((batch_size, 1), device=self.device, dtype=imgs.dtype)

        # noise and labels for generator input
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        gen_labels = torch.randint(0, self.hparams.n_classes, (batch_size,), device=self.device)

        # Generator loss
        gen_imgs = self(z, gen_labels)
        pred_fake = self.discriminator(gen_imgs, gen_labels)
        g_loss = self.adversarial_loss(pred_fake, valid)

        # Discriminator loss
        pred_real = self.discriminator(imgs, labels)
        d_real_loss = self.adversarial_loss(pred_real, valid)
        pred_fake = self.discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = self.adversarial_loss(pred_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2.0

        log_dict = {
            "g_loss": g_loss.detach(),
            "d_loss": d_loss.detach(),
        }

        return log_dict, g_loss, d_loss

    def on_train_epoch_end(self):
        """Generate sample images and log them to WandB (if available) at epoch end."""
        # Skip if no wandb logger is active (avoids wasting GPU time between epochs)
        has_wandb = any(
            type(logger).__name__ == "WandbLogger"
            for logger in self.trainer.loggers
        )
        if not has_wandb:
            return

        try:
            import torchvision
        except Exception:
            return

        n_samples = 16
        z = torch.randn(n_samples, self.hparams.latent_dim, device=self.device)
        labels = torch.arange(0, n_samples, device=self.device) % self.hparams.n_classes

        # Generate samples without building a graph or updating BatchNorm statistics.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            gen_imgs = self(z, labels)
        if was_training:
            self.train()

        # denormalize from [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1.0) / 2.0

        # make grid
        grid = torchvision.utils.make_grid(gen_imgs.cpu(), nrow=4)
        # convert to HWC uint8 numpy array
        img_np = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')

        for logger in self.trainer.loggers:
            if type(logger).__name__ == "WandbLogger":
                try:
                    logger.experiment.log({
                        "gen_imgs": [wandb.Image(img_np, caption=f"epoch_{self.current_epoch}")]
                    }, step=self.global_step)
                except Exception:
                    pass
