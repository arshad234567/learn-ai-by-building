# LLM Weight Compression using LLM Compressor and vLLM

A practical guide to reducing large language model size through post-training quantization while preserving output quality.

---

## Overview

Deploying full-precision language models in production is resource-intensive. Every inference pass consumes significant GPU memory and compute. Quantization addresses this by reducing the numerical precision of model weights, allowing the same model to occupy less memory and run faster — without retraining from scratch.

This guide covers the end-to-end workflow: selecting a base model, picking the right compression algorithm, defining a quantization scheme, measuring size reduction, and validating accuracy using perplexity.

---

## The Four-Step Compression Workflow

Any quantization pipeline using LLM Compressor follows four stages.

The first is model selection. You start with a pretrained model, typically sourced from Hugging Face or an internal model registry. The model's architecture and size influence which algorithms are practical and how much compression is achievable.

The second is algorithm selection. Different algorithms make different tradeoffs between compression speed, memory requirements during calibration, and the accuracy of the resulting model. Choosing the wrong one can waste hours of compute or produce a degraded model.

The third is scheme selection. This means deciding the target precision for weights and activations independently. A common production choice is four-bit weights with sixteen-bit activations, written as W4A16. This preserves the precision of activations while aggressively compressing the stored weights.

The fourth is inference deployment. Once compressed, the model can be loaded directly by inference engines like vLLM, which support quantized formats natively without any additional conversion.

---

## Why Calibration Data Matters

Most quantization algorithms cannot work on weights alone. They need to observe how the model behaves on real inputs to understand which weights are most sensitive to rounding and which can be compressed aggressively without consequence.

This calibration pass uses a small representative dataset — typically a few hundred sequences from a text corpus like WikiText-2. More calibration samples give a more accurate picture of weight importance, but the accuracy gains diminish quickly past a few hundred samples while runtime continues to grow. Two hundred and fifty six samples is a reliable default for most models.

The sequence length used during calibration also matters. Longer sequences let the algorithm observe how weights behave across realistic context windows. Samples that exceed the chosen length are truncated.

---

## Compression Algorithms

**Round-to-Nearest** is the simplest approach. Each weight is rounded to the nearest representable value in the target precision. No calibration data is required and it runs quickly, but accuracy degrades noticeably at four-bit precision. It serves as a useful baseline but is not suitable for production use at lower bit widths.

**AWQ (Activation-Aware Weight Quantization)** is based on the observation that weights are not equally important. Some weights, when changed even slightly, cause large shifts in model outputs. Others can be rounded heavily with little effect. AWQ identifies which is which by examining activation magnitudes during calibration. Weights that correspond to large activations are treated with more precision; the rest are compressed more aggressively. AWQ is computationally lighter than GPTQ, requires less VRAM during calibration, and performs especially well on NVIDIA hardware.

**GPTQ** takes a more mathematically rigorous approach. Rather than just rounding weights, it asks: given the error introduced by rounding this weight, how should the remaining weights be adjusted to compensate so the overall output changes as little as possible? It computes the Hessian of the loss with respect to the weights — a measure of output sensitivity to each specific weight — then works through layers sequentially, quantizing each weight and updating the rest to absorb the introduced error. This process is computationally expensive due to Hessian computation and inversion, but it produces the highest accuracy among the available options and is the most widely supported format for sharing quantized models publicly.

**SparseGPT** handles a different problem — sparsification rather than pure quantization. It is relevant only in specific scenarios involving hardware like the NVIDIA H100 that can exploit structured sparsity efficiently.

Beyond these core algorithms, two preprocessing techniques can be applied before quantization to reduce information loss. Smoothing flattens outlier activation spikes that would otherwise force the quantizer to preserve wide value ranges at the cost of precision. Rotations apply mathematical transformations to weight matrices so that the values are more uniformly distributed and easier to quantize cleanly.

---

## The Reality of Compression Ratios

Going from sixteen-bit to four-bit weights suggests a theoretical four times reduction in model size. In practice, the actual reduction is lower because not all of the model gets quantized.

The linear layers — which contain the vast majority of parameters — are quantized. But the output projection layer that maps internal representations to vocabulary tokens, along with normalization layers and embeddings, remain at higher precision. On a small model like a 0.6B parameter network, these unquantized components represent a meaningful fraction of total size, pulling the overall reduction to roughly forty-two percent rather than seventy-five.

This ratio improves significantly at larger scales. On a 70B parameter model, the linear layers dominate so heavily that the same W4A16 scheme approaches the theoretical compression ceiling much more closely.

---

## Measuring Accuracy Loss with Perplexity

Visual comparison of outputs from the base and compressed models provides some intuition, but production decisions require a number. Perplexity is the standard metric for this purpose.

Perplexity measures how well a language model predicts a held-out sequence of text. It is computed by running the model over the test data, measuring the average cross-entropy loss between the model's predicted token distributions and the tokens that actually appeared, and exponentiating that average. Lower perplexity means the model is less surprised by real text — it is making better predictions.

Evaluation uses the test split of WikiText-2, which is kept separate from the calibration data to prevent any data leakage from artificially flattering the results.

On a 0.6B model with W4A16 quantization, the base model perplexity comes in around 32.79. The quantized model scores approximately 35.48 — roughly an eight percent increase. For most production use cases, this tradeoff is favorable. A few percent of perplexity degradation in exchange for a forty-plus percent reduction in model size and the associated infrastructure savings is a worthwhile exchange.

---

## Choosing the Right Algorithm

Round-to-Nearest is appropriate when you need a fast, rough baseline and do not need production-grade accuracy at low bit widths.

AWQ is the better choice when calibration speed and VRAM efficiency matter, particularly on NVIDIA hardware. It delivers strong accuracy without the heavy computational cost of Hessian-based methods.

GPTQ is the right choice when accuracy retention is the top priority, when you intend to share the quantized model publicly, or when you want the broadest compatibility with inference tooling. It is the industry standard for a reason.

SparseGPT is only relevant when working with hardware specifically designed to exploit sparsity, and is not a general-purpose replacement for the quantization algorithms above.

---

## Key Takeaways

W4A16 quantization compresses model weights to four-bit integers while keeping activations at sixteen-bit precision. The resulting size reduction on small models is around forty percent and grows toward the theoretical four times compression as model scale increases.

Only linear layers are quantized. The output head and normalization layers stay at full precision to protect output quality, which is why real-world compression ratios fall short of the theoretical maximum.

GPTQ produces the best accuracy among available algorithms by compensating for quantization error layer by layer, at the cost of higher memory and compute requirements during the compression pass.

An eight percent increase in perplexity at INT4 precision is generally acceptable for production inference. The infrastructure savings — reduced memory footprint, faster inference, lower hosting cost — outweigh the small accuracy cost in most real-world deployments.

Inference engines like vLLM load quantized models directly, requiring no post-processing or format conversion after the compression step.

---

## License

MIT License.
