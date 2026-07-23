# Pantheon Lab Programming Assignment

## Submission Summary

This repository contains a conditional MNIST GAN implemented with Hydra and PyTorch Lightning. The completed run uses the native 28×28 MNIST resolution, GPU acceleration when available, WandB experiment tracking, and Lightning 2.x manual optimization for the generator/discriminator pair.

### Completed implementation

- Completed the generator and discriminator Hydra configuration.
- Implemented the shared GAN `step()` method and both generator/discriminator losses.
- Added Lightning 2.x manual optimization for the two-optimizer GAN.
- Implemented validation and test steps.
- Added WandB loss logging and generated-image logging at the end of each training epoch.
- Configured WandB as the default logger; use `logger=null` for a run without experiment tracking.
- Added a 20-epoch training horizon and validation every five epochs.
- Added GPU, mixed-precision, native-resolution, batch-size, data-loading, and pinned-memory optimizations.

### Training output

The completed run is tracked in the WandB `Tests` project. The local run artifact is `logs/wandb/run-20260723_202820-r60lag0d/` with run ID `r60lag0d`.

Run summary:

| Item | Result |
|------|--------|
| Hardware | NVIDIA GeForce RTX 5070 Ti |
| Python | 3.14.3 |
| Epochs | 20 (`epoch` 0–19) |
| Global step | 8,599 |
| Runtime | 1,329 seconds |
| Trainable parameters | 2,444,377 |
| Final `train/g_loss` | 0.4273 |
| Final `train/d_loss` | 0.1698 |
| Final `val/g_loss` | 0.5748 |
| Final `val/d_loss` | 0.1634 |

The run records step-level and epoch-level loss curves, epoch-level validation/test metrics, GPU/system metrics, and 16-image conditional sample grids under `gen_imgs`. The final sample grid is captioned `epoch_19` and shows recognizable digits with visible noise, which is expected from this small fully connected GAN. A convolutional DCGAN-style architecture would be the next improvement for sharper samples.

### Challenges and fixes

- **Python 3.14 and Hydra:** upgraded Hydra from the 1.3 release to the Hydra development version from GitHub because of the `LazyCompletionHelp`/`argparse` incompatibility.
- **Hydra composition:** added `_self_` and corrected parent-relative OmegaConf interpolations such as `${..n_classes}`.
- **Lightning 2.x compatibility:** replaced `LightningLoggerBase` with `Logger`, migrated deprecated Trainer options, handled nullable loggers, and used `trainer.loggers` for multiple logger instances.
- **Multiple optimizers:** enabled manual optimization and retained a shared `step()` helper for generator/discriminator loss computation.
- **Windows logging:** configured UTF-8 output to avoid GBK failures when Lightning emits Unicode messages.
- **Runtime behavior:** implemented `test_step`, prevented invalid checkpoint selection during testing, and skipped sample generation when WandB is disabled.

Detailed notes are available in [`challenges.md`](challenges.md) and [`training.md`](training.md).

## Answers

### GAN Model Questions

#### 1. What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

The discriminator is a binary classifier that distinguishes real images from the dataset from fake images produced by the generator. In this implementation it returns a scalar score for each image-label pair. Because the model uses `MSELoss` with least-squares GAN targets, the score is trained toward 1 for real images and 0 for generated images; it is not passed through a sigmoid and should not be interpreted as a calibrated probability.

In this MNIST GAN project, the discriminator takes a 28×28 grayscale image together with its digit label. During training it sees real MNIST images and fake images from the generator. The generator tries to make the discriminator assign generated images a real target score. This adversarial feedback is the core of conditional GAN training.

#### 2. The generator takes `noise` and `labels`. What are these and how are they used to generate the number 5 at inference?

| Input | Description |
|-------|-------------|
| `noise` | Random latent vector (∼ standard normal distribution). Provides randomness so each run produces a different-looking digit. |
| `labels` | Conditional class label (0–9) telling the generator which digit to produce. Passed through an embedding or one-hot encoding. |

**To generate the number 5 at inference time:**
  1. Sample a noise vector from N(0, 1) (e.g., 100 dimensions).
  2. Set the label to 5.
  3. Feed both noise and label into the generator.
  4. The generator outputs a 28×28 image of the digit "5".

This is the standard inference procedure for a **Conditional GAN (CGAN)**, where the label controls the class of the generated output.

#### 3. What steps are needed to deploy a model into production?

| Step | Description |
|------|-------------|
| Model Export | Convert the trained model to a deployment-friendly format (TorchScript, ONNX, or save as a checkpoint). |
| Inference Optimization | Apply quantization, pruning, or use acceleration frameworks like TensorRT or vLLM to reduce latency. |
| API Wrapping | Expose the model via a REST API (FastAPI/Flask) or gRPC for high-performance serving. |
| Containerization | Package the application with Docker to ensure environment consistency across dev/staging/prod. |
| Deployment | Deploy to Kubernetes, AWS Lambda, or cloud VMs with auto-scaling configured. |
| Monitoring & Logging | Track inference latency, throughput, error rates, and set up alerts. |
| Versioning | Use MLflow or DVC for model version control, enabling A/B testing and rollback. |

#### 4. When training with multiple GPUs, how do you ensure data is allocated to the correct GPU in PyTorch Lightning?

PyTorch Lightning handles most device placement automatically, but here are the key points:

| Area | How Lightning Handles It |
|------|--------------------------|
| Distributed Init | Set `accelerator="gpu"`, `devices=N`, and an appropriate distributed strategy such as `strategy="ddp"`. Lightning initializes the distributed process group. |
| Model & Loss | `self` (the model) is moved to the correct device automatically. No manual `.to(device)` calls needed. |
| Data | In `training_step`, the `batch` is already on the correct GPU. |
| Manual Tensors | Create tensors on the fly using `self.device`: `torch.randn(batch_size, latent_dim, device=self.device)` |
| Cross-GPU Sync | Use `self.all_gather()` to aggregate tensors across multiple GPUs. |

The main advantage of Lightning is that it abstracts away low-level distributed training details, letting you focus on the research logic rather than device management.

### LLM Questions

#### 1. Compare at least 3 different models on Content Quality, Contextual Understanding, Language Fluency, and Ethical Considerations.

> **Remaining manual deliverable:** Run the same prompts against three available models on an evaluation platform such as Poe, record the model versions and settings, and add the responses and comparison below. Results should not be fabricated or inferred from model names alone.

Recommended evaluation protocol:

1. Use the same system instruction, user prompt, temperature, and maximum output length for every model.
2. Test factual summarization, ambiguous context resolution, structured writing, and a safety-sensitive prompt.
3. Save the raw responses and record the date, model version, parameters, and any platform-specific system prompt.
4. Compare each model on content quality, contextual understanding, language fluency, and ethical behavior using concrete response excerpts.
5. Repeat prompts where outputs are stochastic, or use temperature 0 when deterministic comparison is more important.

#### 2. What parameters can be used to control model responses? Explain in detail.

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **Temperature** | Scales logits before softmax. Range: 0.0 – 2.0 (typically). | Low (0.0–0.3): deterministic, repetitive. High (0.7–1.5): creative, diverse, but may hallucinate. |
| **Top-p** (Nucleus Sampling) | Only samples from the smallest set of tokens whose cumulative probability ≥ `p`. Range: 0.0 – 1.0. | Low (0.1): focused, safe. High (0.9): more varied outputs. Commonly paired with temperature. |
| **Top-k** | Limits sampling to the `k` most likely next tokens. | Low `k`: constrained, repetitive. High `k`: more diverse. Less commonly used now vs. top-p. |
| **Max Tokens** | Caps the total output length (input + output count toward the model's context window). | Prevents runaway responses and controls cost. Too low may truncate meaningful output. |
| **Stop Sequences** | String(s) that cause generation to halt immediately when encountered. | Useful for structured outputs (e.g., stop at `\n\n` for single-paragraph responses, or at `###` for section breaks). |
| **Frequency Penalty** | Penalizes tokens based on how often they've appeared so far. Range: -2.0 – 2.0. | Positive values reduce repetition. Negative values encourage repeated terms. |
| **Presence Penalty** | Penalizes tokens that have appeared at all (regardless of frequency). Range: -2.0 – 2.0. | Positive values encourage the model to talk about new topics rather than repeating itself. |

#### 3. Explore prompt engineering techniques: template-based, rule-based, and ML-based prompts. Include challenges and examples.

| Technique | Description | Example | Challenges |
|-----------|-------------|---------|------------|
| **Template-Based** | Predefined prompt structures with placeholders filled dynamically. | `"Summarize the following article in 3 bullet points:\n\n{article_text}"` | Templates can be too rigid for complex tasks. Requires anticipating all input variations upfront. Poor templates produce low-quality outputs. |
| **Rule-Based** | Prompts that include explicit instructions, constraints, or formatting rules. | `"Answer only 'Yes' or 'No'. Do not explain.\n\nQuestion: Is the sky blue?"` | Rules may be ignored by the model if not emphatic enough. Conflicting rules confuse the model. Needs iterative testing to get rules right. |
| **ML-Based (Soft Prompting)** | Trainable prompt embeddings optimized via gradient descent (e.g., prefix tuning, P-tuning). | Fine-tuning a small set of virtual tokens prepended to the input, while keeping the LLM frozen. | Requires training data and compute. Less interpretable — hard to debug why a prompt embedding works. May not generalize across tasks. |

**Key considerations when designing prompts:**
- **Clarity** — Be explicit about the desired format, tone, and scope.
- **Specificity** — Vague prompts produce vague answers. Include concrete constraints.
- **Context relevance** — Provide relevant background without overwhelming the model.
- **Iterative refinement** — Rarely works on the first try. Test, evaluate, and iterate.
- **Bias awareness** — Prompts can inadvertently steer the model toward biased responses.

#### 4. What is Retrieval-Augmented Generation (RAG) and how is it applied in NLG tasks?

**RAG** is a hybrid architecture that combines a **retriever** (searches a knowledge base for relevant documents) with a **generator** (an LLM that produces the final response conditioned on those documents).

| Component | Role |
|-----------|------|
| Retriever | Indexes a corpus (documents, FAQs, databases). At query time, fetches the top-k most relevant passages via semantic search (embeddings) or keyword search (BM25). |
| Generator | Receives the user query + retrieved passages as context, then generates a grounded, factual response. |

**Applications in NLG tasks:**

| Task | How RAG Helps |
|------|---------------|
| **Question Answering** | Retrieves relevant documents before answering, reducing hallucination and improving factual accuracy. |
| **Summarization** | Fetches source materials to produce summaries grounded in actual content rather than model memory. |
| **Dialogue / Chatbots** | Pulls from knowledge bases, product docs, or internal wikis to answer domain-specific queries the base model was never trained on. |
| **Code Generation** | Retrieves relevant documentation, API references, or similar code examples before generating solutions. |

RAG's key advantage: it decouples **knowledge** (stored in the retriever's index, easily updatable) from **reasoning** (handled by the LLM), enabling factual, current, and domain-specific responses without fine-tuning.

<div align="center">

## Original Project Description

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## What is all this?
This "programming assignment" is really just a way to get you used to
some of the tools we use every day at Pantheon to help with our research.

There are 4 fundamental areas that this small task will have you cover:

1. Getting familiar with training models using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

2. Using the [Hydra](https://hydra.cc/) framework

3. Logging and reporting your experiments on [weights and biases](https://wandb.ai/site)

4. Showing some basic machine learning knowledge

## What's the task?
The actual machine learning task you'll be doing is fairly simple! 
You will be using a very simple GAN to generate fake
[MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) images.

We don't excpect you to have access to any GPU's. As mentioned earlier this is just a task
to get you familiar with the tools listed above, but don't hesitate to improve the model
as much as you can!

## What you need to do

To understand how this framework works have a look at `src/train.py`. 
Hydra first tries to initialise various pytorch lightning components: 
the trainer, model, datamodule, callbacks and the logger.

To make the model train you will need to do a few things:

- [x] Complete the model yaml config (`model/mnist_gan_model.yaml`)
- [x] Complete the implementation of the model's `step` method
- [x] Implement logging functionality to view step/epoch loss curves and generated samples during training using the Lightning `on_train_epoch_end` hook and [Weights & Biases](https://wandb.ai/site). WandB is enabled by default; use `logger=null` to disable it.
- [x] Answer the GAN and deployment questions below
- [ ] Run and document the external three-model LLM comparison below

**All implementation tasks in the code are marked with** `TODO`

Don't feel limited to these tasks above! Feel free to improve on various parts of the model

For example, training the model for around 20 epochs will give you results like this:

![example_train](./images/example_train.png)

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name pantheon-py38 python=3.8
conda activate pantheon-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default (GPU)
python run.py experiment=train_mnist_gan.yaml

# train on CPU
python run.py experiment=train_mnist_gan.yaml trainer.accelerator=cpu

# train on GPU
python run.py experiment=train_mnist_gan.yaml trainer.accelerator=gpu trainer.devices=1
```

You can override any parameter from command line like this
```yaml
python run.py experiment=train_mnist_gan.yaml trainer.max_epochs=20 datamodule.batch_size=32
```

The generator and discriminator are configured in `configs/model/mnist_gan_model.yaml`. The model uses Lightning 2.x manual optimization because GAN training requires two optimizers.

## Open-Ended tasks (Bonus for junior candidates, expected for senior candidates)

Staying within the given Hydra - Pytorch-lightning - Wandb framework, show off your skills and creativity by extending the existing model, or even setting up a new one with completely different training goals/strategy. Here are a few potential ideas:

- **Implement your own networks**: you are free to choose what you deem most appropriate, but we recommend using CNN and their variants if you are keeping the image-based GANs as the model to train
- **Use a more complex dataset**: ideally introducing color, and higher resolution
- **Introduce new losses, or different training regimens**
- **Add more plugins/dependecy**: on top of the provided framework
- **Train a completely different model**: this may be especially relevant to you if your existing expertise is not centered in image-based GANs. You may want to re-create a toy sample related to your past research. Do remember to still use the provided framework.

## Questions

Try to prepare some short answers to the following questions below for discussion in the interview.

* What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

* The generator network in this code base takes two arguments: `noise` and `labels`.
What are these inputs and how could they be used at inference time to generate an image of the number 5?

* What steps are needed to deploy a model into production?

* If you wanted to train with multiple GPUs, 
what can you do in pytorch lightning to make sure data is allocated to the correct GPU? 

## Submission

- Using git, keep the existing git history and add your code contribution on top of it. Follow git best practices as you see fit. We appreciate readability in the commits
- Add a section at the top of this README, containing your answers to the questions, as well as the output `wandb` graphs and images resulting from your training run. You are also invited to talk about difficulties you encountered and how you overcame them
- Link to your git repository in your email reply and share it with us/make it public

# Chatbot Assignment:

To complete this assignment, please use any LLM evaluation platform or tool you are familiar with — or simply try with [Poe](https://poe.com/) — to test different models, capture their responses, and document your findings.

* Compare atleast 3 different models and provide insights on Content Quality, Contextual Understanding, Language Fluency and Ethical Considerations with examples.

* What are the parameters that can be used to control response. Explain in detail.

* Explore various techniques used in prompt engineering, such as template-based prompts, rule-based prompts, and machine learning-based prompts and provide what are the challenges and considerations in designing effective prompts with examples.

* What is retrieval-augmented generation(RAG) and how is it applied in natural language generation tasks?

<br>
