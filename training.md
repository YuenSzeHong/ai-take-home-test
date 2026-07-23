# Training Improvements

Documenting steps taken to improve training speed and stability.

![Generated MNIST samples from WandB epoch 19](images/generated_epoch_19.png)

---

## 1. GPU Acceleration

**Before:** `accelerator: cpu` — training on CPU.

**After:** `accelerator: gpu` with `devices: 1`. The RTX 5070 Ti is orders of magnitude faster for GAN training.

```yaml
# configs/trainer/default.yaml
accelerator: gpu
devices: 1
```

---

## 2. Multi-Process Data Loading

**Before:** `num_workers: 0` — single-process loading. GPU sits idle waiting for next batch.

**After:** `num_workers: 4` — subprocess workers pre-fetch batches in parallel with training.

```yaml
# configs/datamodule/mnist_datamodule.yaml
num_workers: 4
```

---

## 3. Native MNIST Resolution

**Before:** `img_size: 32` — every 28×28 MNIST image was upscaled via `transforms.Resize(32)`, wasting compute on both the data pipeline and model forward passes.

**After:** `img_size: 28` — matches native MNIST resolution. No unnecessary interpolation.

```yaml
# configs/model/mnist_gan_model.yaml
img_size: 28

# configs/datamodule/mnist_datamodule.yaml
img_size: 28
```

---

## 4. Tensor Cores (TF32)

**Before:** Default float32 precision, underutilizing the RTX 5070 Ti's Tensor Cores.

**After:** `torch.set_float32_matmul_precision("high")` enables TF32 mode on matmul operations, trading negligible precision for significant throughput. Guarded with `torch.cuda.is_available()` to stay compatible with CPU-only environments.

```python
# run.py (inside main())
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
```

---

## 5. Fixed Training Horizon

GAN losses oscillate by design, so `val/g_loss` is not a reliable early-stopping signal. The default run uses a fixed 20-epoch horizon, matching the README's example, and generated samples are inspected in WandB to judge image quality.

```yaml
# configs/trainer/default.yaml
max_epochs: 20
check_val_every_n_epoch: 5
```

---

## 6. Logging Quality

Training logging now records both per-step and epoch-aggregated metrics:

- `train/g_loss` and `train/d_loss` are logged per step and averaged per epoch.
- `val/g_loss` and `val/d_loss` are logged as epoch-level metrics.
- `test/g_loss` and `test/d_loss` are logged as epoch-level metrics.
- Generated samples are logged to WandB once per training epoch at the current Lightning global step.
- Sample generation uses evaluation mode and `torch.no_grad()`, so it does not update BatchNorm statistics or build an unnecessary gradient graph.
- When no WandB logger is configured, sample generation is skipped entirely.

This gives WandB both detailed training behavior and smoother epoch-level curves. The per-step values are useful for diagnosing GAN instability; the epoch values are better for comparing overall progress.

## Further Ideas (Not Yet Implemented)

| Idea | Benefit | Trade-off |
|------|---------|-----------|
| Mixed precision (`torch.float16`) | 2× faster on Tensor Core GPUs | May destabilize GAN training |
| Gradient accumulation | Simulate larger batch sizes | Slower convergence per step |
| Spectral normalization | Stabilize discriminator | Extra compute per forward pass |
| Learning rate scheduling | Better convergence | More hyperparameters to tune |
| `pin_memory: True` | Faster CPU→GPU transfer | Uses more pinned RAM |
