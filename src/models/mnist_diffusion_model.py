"""Conditional DDPM for MNIST digit generation."""

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule

import wandb

from src.models.modules.unet import ConditionalUNet


class MNISTDiffusionModel(LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        n_classes: int = 10,
        img_size: int = 28,
        channels: int = 1,
        timesteps: int = 200,
        base_channels: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.timesteps = timesteps
        self.img_size = img_size

        self.model = ConditionalUNet(
            in_channels=channels,
            out_channels=channels,
            n_classes=n_classes,
            base_channels=base_channels,
        )

        # Precompute DDPM schedule
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        return self.model(x, t, labels)

    def _noise_batch(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample t and noise, return x_t, noise, t."""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise, t

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x0, labels = batch
        x_t, noise, t = self._noise_batch(x0)
        pred = self(x_t, t, labels)
        loss = self.criterion(pred, noise)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        x0, labels = batch
        x_t, noise, t = self._noise_batch(x0)
        pred = self(x_t, t, labels)
        loss = self.criterion(pred, noise)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        x0, labels = batch
        x_t, noise, t = self._noise_batch(x0)
        pred = self(x_t, t, labels)
        loss = self.criterion(pred, noise)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, n_samples: int, labels: Tensor) -> Tensor:
        """Generate n_samples images for the given labels via DDPM sampling."""
        self.eval()
        x = torch.randn(n_samples, self.hparams.channels, self.img_size, self.img_size, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            eps_pred = self(x, t_batch, labels)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred
            ) + torch.sqrt(beta_t) * z
        return x

    def on_train_epoch_end(self) -> None:
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
        labels = torch.arange(0, n_samples, device=self.device) % self.hparams.n_classes
        imgs = self.sample(n_samples, labels)
        imgs = (imgs + 1.0) / 2.0
        grid = torchvision.utils.make_grid(imgs.cpu(), nrow=4)
        img_np = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")

        for logger in self.trainer.loggers:
            if type(logger).__name__ == "WandbLogger":
                try:
                    logger.experiment.log(
                        {
                            "diffusion_samples": [
                                wandb.Image(
                                    img_np, caption=f"epoch_{self.current_epoch}"
                                )
                            ]
                        },
                        step=self.global_step,
                    )
                except Exception:
                    pass
