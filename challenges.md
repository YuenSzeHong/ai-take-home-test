# Challenges & Fixes

Documenting issues encountered while setting up this project and how they were resolved.

---

## 1. Python 3.14 + Hydra 1.3: `LazyCompletionHelp` Type Error

**Symptom:**
```
TypeError: argument of type 'LazyCompletionHelp' is not a container or iterable
ValueError: badly formed help string
```

**Cause:** Python 3.14 changed `argparse._check_help()` to use `'%' not in help_string`, which fails when `help_string` is Hydra's `LazyCompletionHelp` object (not a plain string). Hydra 1.3.x does not support Python 3.14.

**Fix:** Upgraded to Hydra 1.4 (development) from GitHub, which natively supports Python 3.10–3.14:
```bash
uv pip install git+https://github.com/facebookresearch/hydra.git
```

*Note: A monkey-patch for `argparse.ArgumentParser._check_help` was used as a temporary workaround before the upgrade.*

---

## 2. Hydra 1.4: Missing `_self_` in Defaults List

**Symptom:**
```
UserWarning: In 'config.yaml': Defaults list is missing `_self_`.
```

**Cause:** Hydra 1.2+ requires `_self_` to be explicitly placed in the defaults list to control where the config's own values merge in the composition order.

**Fix:** Added `- _self_` at the end of the defaults list in `configs/config.yaml`.

---

## 3. PyTorch Lightning 2.0+: `LightningLoggerBase` Renamed

**Symptom:**
```
ImportError: cannot import name 'LightningLoggerBase' from 'pytorch_lightning.loggers'
```

**Cause:** PyTorch Lightning ≥2.0 renamed `LightningLoggerBase` to `Logger`.

**Fix:** Replaced all references to `LightningLoggerBase` with `Logger` in:
- `src/train.py` (import + type annotation)
- `src/utils/utils.py` (type annotations in `log_hyperparameters` and `finish`)

---

## 4. OmegaConf: Stricter Interpolation Path Resolution

**Symptom:**
```
omegaconf.errors.InterpolationKeyError: Interpolation key 'n_classes' not found
    full_key: model.generator.n_classes
```

**Cause:** The newer OmegaConf version bundled with Hydra 1.4 enforces stricter relative interpolation resolution. `${n_classes}` inside `model.generator` resolves relative to the current node (`model.generator.n_classes`) rather than the parent (`model.n_classes`).

**Fix:** Changed relative interpolations to parent-relative paths using `..`:
- `configs/model/mnist_gan_model.yaml`: `${n_classes}` → `${..n_classes}` (and same for `latent_dim`, `channels`, `img_size`)
- `configs/datamodule/mnist_datamodule.yaml`: `${data_dir}` → `${..data_dir}`

---

## 5. PyTorch Lightning 2.0+: Deprecated Trainer Arguments

**Symptom:**
```
TypeError("Trainer.__init__() got an unexpected keyword argument 'gpus'")
```

**Cause:** PyTorch Lightning 2.0 removed several Trainer constructor arguments:
- `gpus` → replaced by `accelerator` + `devices`
- `progress_bar_refresh_rate` → removed (progress bar always on)
- `weights_summary` → replaced by `enable_model_summary`
- `resume_from_checkpoint` → moved to `trainer.fit(ckpt_path=...)`

**Fix:** Updated `configs/trainer/default.yaml` with the new 2.0 API. Also updated the debug-mode GPU check in `src/utils/utils.py` to use `accelerator` instead of `gpus`.

---

## 6. PyTorch Lightning 2.0+: `trainer.logger` Can Be `None`

**Symptom:**
```
AttributeError: 'NoneType' object has no attribute 'log_hyperparams'
```

**Cause:** In Lightning 2.0, when no loggers are configured (`logger: null`), `trainer.logger` is `None` rather than a dummy logger. The `log_hyperparameters` function in `utils.py` assumed a logger was always present.

**Fix:** Added `if trainer.logger:` guard before calling `trainer.logger.log_hyperparams` in `src/utils/utils.py`.

---

## 7. PyTorch Lightning 2.0+: `trainer.logger` (singular) vs `trainer.loggers` (plural)

**Symptom:** Wandb image logging in `on_epoch_end` would silently fail.

**Cause:** Lightning 2.0 changed `trainer.logger` from a list to a single logger (or `None`). The plural `trainer.loggers` is always a list.

**Fix:** Changed `for logger in self.trainer.logger:` → `for logger in self.trainer.loggers:` in `src/models/mnist_gan_model.py`.

---

## 8. Test Step `NotImplementedError`

**Symptom:** Training completes but crashes on test with `NotImplementedError`.

**Cause:** `test_step` raised `NotImplementedError` but `test_after_training: True` in config.

**Fix:** Implemented a minimal `test_step` in `src/models/mnist_gan_model.py` that mirrors the validation step, logging test losses.

---

## 9. Windows: Unicode Emoji Logging Crash

**Symptom:**
```
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f4a1' in position 73
```

**Cause:** Python's default stdout encoding on Windows is GBK, which can't handle emoji characters like 💡 in Lightning's log messages.

**Fix:** Reconfigured stdout/stderr to UTF-8 at the top of `run.py`:
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

---

## 10. PyTorch Lightning 2.0+: Multiple Optimizers Require Manual Optimization

**Symptom:**
```
RuntimeError: Training with multiple optimizers is only supported with manual optimization.
```

**Cause:** Lightning 2.0 dropped automatic optimization with multiple optimizers. The `optimizer_idx` argument in `training_step` is no longer supported.

**Fix:** Rewrote `src/models/mnist_gan_model.py` to use manual optimization:
- Set `self.automatic_optimization = False`
- Keep loss computation in the shared `step()` method
- In `training_step`, manually call `opt.zero_grad()`, `self.manual_backward(loss)`, and `opt.step()` for each optimizer
- Use the shared loss helper from `training_step`, `validation_step`, and `test_step`

---

## 11. MNIST Image Size: 32 → 28

**Symptom:** Training slower than expected; unnecessary upscaling.

**Cause:** The template defaulted to `img_size: 32` (powers of 2 are common for GANs), but MNIST images are natively 28×28. Every image was being upscaled via `transforms.Resize(32)`, wasting compute.

**Fix:** Changed `img_size` to 28 in both `configs/model/mnist_gan_model.yaml` and `configs/datamodule/mnist_datamodule.yaml` to match the native MNIST resolution. Also bumped `num_workers` from 0 to 4 for faster data loading.

---

## 12. WandB Runs Remaining Open After Errors

**Symptom:** A run could remain marked as running when training or testing raised an exception.

**Cause:** Logger finalization was only reached after successful training and testing.

**Fix:** Wrapped the training and test phases in `try`/`except`/`finally`, report `success` or `failed`, and call each Lightning logger's `finalize(status)` from the `finally` block.

---

## 15. WandB Shutdown Hung During Upload Retry

**Symptom:** The training process reached successful finalization but remained in the terminal while WandB retried an HTTP 500 upload timeout.

**Cause:** `wandb.finish()` waits for pending uploads. A slow or unavailable WandB backend can keep retrying during shutdown.

**Fix:** Set `wandb.Settings(finish_timeout=60)` in `configs/logger/wandb.yaml` and call the WandB SDK directly before any generic Lightning logger finalization. This ensures the timeout applies; any unsent local data remains available for `wandb sync`.

---

## 13. PyTorch 2.6: Testing Accidentally Restored an Old Checkpoint

**Symptom:** Testing failed with `Weights only load failed` while loading an existing checkpoint.

**Cause:** Lightning can automatically select a checkpoint when testing a connected model. Old checkpoints contain pickled model classes that PyTorch 2.6 rejects with its safer `weights_only=True` default.

**Fix:** Pass the current in-memory `model` and `datamodule` explicitly to `trainer.test(..., ckpt_path=None)`, preventing Lightning from restoring an old checkpoint for the post-training test.

---

## 14. WandB Run Did Not Receive a Completion Signal

**Symptom:** The WandB panel continued to show a run as active after local training had ended or failed.

**Cause:** Lightning logger finalization alone did not reliably flush and close the WandB SDK run in this environment. In addition, cleanup could be skipped when `fit()` or `test()` raised an exception.

**Fix:** Always finalize loggers from a `finally` block and explicitly call `wandb.finish(exit_code=...)` for WandB runs. Successful runs use exit code 0; failed runs use exit code 1.
