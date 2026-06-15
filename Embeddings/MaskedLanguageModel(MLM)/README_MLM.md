# Masked Language Modeling

This document explains the concept of Masked Language Modeling, how it works as a pre-training objective, how BERT implements it, and how a fine-tuned MLM model is used for mask prediction inference.

---

## Table of Contents

1. [What is Masked Language Modeling](#1-what-is-masked-language-modeling)
2. [Why MLM Exists](#2-why-mlm-exists)
3. [How Masking Works](#3-how-masking-works)
4. [BERT and the MLM Objective](#4-bert-and-the-mlm-objective)
5. [Tokenization and Input Preparation](#5-tokenization-and-input-preparation)
6. [The Data Collator](#6-the-data-collator)
7. [Model Architecture for MLM](#7-model-architecture-for-mlm)
8. [Training Configuration](#8-training-configuration)
9. [Inference: Predicting Masked Tokens](#9-inference-predicting-masked-tokens)
10. [Saving and Reloading a Trained Model](#10-saving-and-reloading-a-trained-model)
11. [Limitations and Practical Considerations](#11-limitations-and-practical-considerations)

---

## 1. What is Masked Language Modeling

Masked Language Modeling is a self-supervised pre-training objective for language models. In this task, a portion of the tokens in an input sequence are replaced with a special placeholder token, and the model is trained to predict the original tokens that were masked.

The key property that makes this useful is that it is self-supervised. No human-labeled data is needed. Any raw text corpus can be used as training data because the labels are derived automatically from the text itself by hiding tokens and using the originals as targets.

The result of training on this objective is a model whose internal representations capture rich contextual information about language, because predicting a missing word accurately requires understanding the surrounding words, their grammatical roles, and their semantic relationships.

---

## 2. Why MLM Exists

Before MLM, language models were typically trained with a left-to-right autoregressive objective. Given a sequence of words, the model predicted the next word. This is effective but has a directional constraint: the model can only use past context when building a representation of the current token.

MLM removes this constraint. Because the model must predict a masked token using both the tokens to its left and the tokens to its right, it is forced to build bidirectional representations. Every token in the output sequence is informed by the full input context. This bidirectionality is one of the central reasons BERT-style models outperformed earlier unidirectional language models on a wide range of downstream tasks.

---

## 3. How Masking Works

Given a tokenized input sequence, a fixed proportion of token positions are selected for masking. The standard proportion used in BERT is 15 percent of all tokens in the sequence.

Of the selected positions, three different substitutions are applied with different probabilities to prevent the model from simply learning to detect the mask token rather than learning meaningful contextual representations:

- 80 percent of the time, the selected token is replaced with the special [MASK] token.
- 10 percent of the time, the selected token is replaced with a random token from the vocabulary.
- 10 percent of the time, the selected token is left unchanged.

This mixed strategy is important. If every selected token were replaced with [MASK], the model would learn to only make predictions when it sees that specific token and would not develop representations useful for tokens that appear normally. By sometimes replacing with a random token or leaving the token unchanged, the model must always maintain a useful representation of every token, because it cannot be certain which tokens have been altered.

The loss is computed only over the positions that were originally selected for masking. Tokens that were not selected do not contribute to the loss regardless of whether they were altered.

---

## 4. BERT and the MLM Objective

BERT (Bidirectional Encoder Representations from Transformers) was the model that popularized MLM as a pre-training objective. BERT is a Transformer encoder that processes the entire input sequence simultaneously using self-attention, allowing information to flow in both directions between all token pairs.

BERT-base-uncased, the model used in this notebook, has the following architecture:

- A vocabulary of 30,522 tokens
- Hidden size of 768 dimensions
- 12 Transformer encoder layers
- 12 attention heads per layer
- An intermediate feed-forward size of 3,072 in each layer

The uncased variant lowercases all input text before tokenization, meaning the model treats "Apple" and "apple" as the same token. This is appropriate for tasks where case does not carry semantic meaning.

BERT was originally pre-trained on BookCorpus and English Wikipedia, amounting to approximately 3.3 billion words. The MLM objective ran alongside a second objective called Next Sentence Prediction. Later research showed that the MLM objective accounts for most of the representational quality, while NSP is less critical.

When loading a pre-trained BERT checkpoint for MLM specifically, a prediction head is attached on top of the encoder. This head takes the 768-dimensional hidden state of each masked position and projects it through a dense layer, a GELU activation, a layer normalization, and a final linear decoder layer back up to the vocabulary size of 30,522. The output is a distribution over all vocabulary tokens for each masked position.

---

## 5. Tokenization and Input Preparation

Before text can be fed to the model, it must be converted into token IDs. The tokenizer for BERT-base-uncased uses WordPiece tokenization, which breaks words into subword units drawn from a learned vocabulary of 30,522 entries.

WordPiece handles out-of-vocabulary words gracefully by decomposing them into known subword fragments. For example, an uncommon word like "tokenization" might be split into "token", "##ization". The "##" prefix indicates that the fragment is a continuation of the previous token rather than the start of a new word.

Each tokenized sequence is prepared with two special tokens added by the tokenizer automatically: a [CLS] token prepended to the start and a [SEP] token appended to the end. [CLS] is used as an aggregate representation in classification tasks. [SEP] marks the boundary between segments.

Three tensors are produced for each input:

**input_ids** contains the integer ID of each token in the sequence, including special tokens and padding tokens where the sequence is shorter than the maximum length.

**attention_mask** is a binary tensor with a 1 at every position that contains a real token and a 0 at every padding position. The model uses this to ignore padding when computing attention.

**token_type_ids** distinguishes tokens belonging to the first segment from tokens belonging to the second segment in tasks involving sentence pairs. For single-sentence inputs, all values are 0.

Sequences are padded or truncated to a fixed maximum length to allow batching. In this notebook the maximum length is set to 128 tokens.

---

## 6. The Data Collator

The DataCollatorForLanguageModeling handles the dynamic application of masking at training time rather than masking the dataset statically in advance. This means that each time the model sees a given example during training, a different random set of tokens is selected for masking, effectively providing data augmentation.

The collator receives a batch of tokenized examples and applies the 15 percent masking probability to each sequence. It handles all three substitution types: masking with [MASK], replacing with a random token, and leaving unchanged. It also constructs the labels tensor, which contains the original token IDs at masked positions and a sentinel value of -100 at all other positions. The loss function ignores positions with label -100, so only the masked positions contribute to the training signal.

---

## 7. Model Architecture for MLM

The full model for masked language modeling consists of two components stacked together.

The base encoder is the BertModel, which is the core stack of 12 Transformer layers. This component takes token IDs as input, converts them to embeddings, adds positional embeddings and token type embeddings, and then passes the combined representation through 12 sequential self-attention and feed-forward layers. The output is a 768-dimensional hidden state for every token position in the input.

The prediction head is BertOnlyMLMHead, which is a lightweight module placed on top of the encoder. It receives the hidden states from the final encoder layer and projects them back to vocabulary size for every position. During training, only the predictions at masked positions are used to compute the loss.

The pooler layer present in the original BERT checkpoint (used for NSP) is not used in the MLM-only setup, which is why the warning about unused weights appears when loading the checkpoint. This is expected behavior and not an error.

---

## 8. Training Configuration

The TrainingArguments object controls every aspect of the training loop. The key hyperparameters used in this notebook are:

**num_train_epochs** is set to 5, meaning the model makes 5 complete passes over the training dataset. For a small dataset with only 5 examples, this is necessary to allow the model to update its weights a meaningful number of times, though in practice MLM pre-training uses far more data and more epochs.

**per_device_train_batch_size** is set to 8. Since the dataset has only 5 examples, the effective batch size per step is 5 rather than 8 because the dataset is smaller than the batch size.

**learning_rate** is set to 5e-5, which is a standard value for fine-tuning BERT-scale models. This is lower than typical training from scratch to avoid disrupting the pre-learned representations.

**weight_decay** is set to 0.01. Weight decay is a regularization technique that adds a penalty proportional to the magnitude of the model's weights to the loss. This discourages the model from assigning very large values to individual parameters and helps prevent overfitting.

**fp16** enables mixed-precision training on CUDA devices. With fp16, computations are done in 16-bit floating point where possible, which reduces memory usage and speeds up training on compatible GPUs. On CPU, this flag is set to false automatically.

**save_strategy** set to epoch saves a checkpoint of the model at the end of every training epoch.

**report_to** set to none disables integration with experiment tracking tools such as Weights and Biases.

---

## 9. Inference: Predicting Masked Tokens

After training, the model can be used to fill in masked positions in new input text. The inference process follows these steps.

First, the input sentence containing the [MASK] token is tokenized with the same tokenizer used during training. The resulting tensors are moved to the same device as the model.

Second, the model is run in evaluation mode with gradient computation disabled. The output contains logits of shape (batch_size, sequence_length, vocabulary_size). Each position in the sequence has a score for every token in the vocabulary.

Third, the position of the [MASK] token in the input_ids tensor is located. This gives the index into the sequence dimension of the logits.

Fourth, the logits at the mask position are extracted. These are a vector of 30,522 values representing the model's confidence in each vocabulary token being the correct fill for the masked position.

Fifth, the top-k tokens are identified by selecting the indices with the highest logit values. The top-5 tokens represent the model's five most confident predictions for what belongs in the masked position.

Finally, the token IDs are decoded back into human-readable words using the tokenizer, and the best prediction is substituted into the original sentence to produce the completed text.

---

## 10. Saving and Reloading a Trained Model

After training, both the model weights and the tokenizer are saved to disk. Saving the tokenizer alongside the model is important because inference requires the same vocabulary and tokenization rules that were used during training. Loading the model without the correct tokenizer would produce incorrect token IDs and unreliable predictions.

The Trainer's save_model method writes the model weights and configuration to the specified directory. The tokenizer's save_pretrained method writes the vocabulary file, tokenizer configuration, and any special token mappings. Both can be reloaded from the same directory using the respective from_pretrained methods.

---

## 11. Limitations and Practical Considerations

**Dataset size**: The notebook uses only 5 training sentences, which is far too small to meaningfully alter the pre-trained weights of a 110 million parameter model. In practice, domain-adaptive pre-training with MLM requires at minimum thousands of sentences and ideally hundreds of thousands or more to shift the model's representations toward a target domain.

**Pre-trained starting point**: Because the notebook starts from a pre-trained BERT checkpoint rather than random initialization, the model already has strong general language representations. Fine-tuning on 5 examples for 5 epochs will produce very small weight updates and the model's behavior will remain close to the original pre-trained model.

**Static masking vs dynamic masking**: Applying the data collator at training time implements dynamic masking, where different tokens are masked on each pass over the data. This is preferable to static masking, where the masked positions are fixed before training, because it exposes the model to more variation across epochs on small datasets.

**Inference with pipeline**: For production use, the Hugging Face pipeline abstraction wraps all tokenization, inference, and decoding steps into a single call and handles multiple mask positions, top-k filtering, and probability scoring automatically. The step-by-step inference shown in this notebook illustrates the underlying mechanics.

**Domain adaptation**: One of the most practical uses of MLM fine-tuning on a domain-specific corpus is to adapt a general-purpose model to specialized vocabulary. If the target domain uses technical terminology that rarely appeared in the original pre-training corpus, running MLM training on in-domain text updates the token representations to better reflect domain-specific usage patterns before fine-tuning on a labeled downstream task.

---

## Glossary

**Masked Language Modeling (MLM)** - A self-supervised pre-training objective where a subset of input tokens are replaced with a mask token and the model is trained to predict the originals.

**Self-supervised learning** - A training paradigm where labels are derived automatically from the input data itself, requiring no human annotation.

**Bidirectional encoding** - A representation strategy where each token's vector is computed using both left and right context simultaneously, as opposed to left-to-right or right-to-left only.

**WordPiece tokenization** - A subword tokenization algorithm that splits words into units drawn from a learned vocabulary, handling unknown words by decomposing them into known fragments.

**[MASK] token** - A special vocabulary token used to replace tokens selected for prediction during MLM training.

**[CLS] token** - A special token prepended to every input sequence, whose final hidden state is often used as an aggregate sequence representation for classification.

**[SEP] token** - A special token appended to each input segment to mark its boundary.

**Data collator** - A component that assembles individual examples into batches and applies dynamic transformations such as masking at training time.

**Logits** - The raw unnormalized scores output by the model before applying softmax. Higher logits correspond to higher predicted probability after normalization.

**Weight decay** - A regularization technique that penalizes large parameter values by adding a term proportional to parameter magnitude to the training loss.

**Mixed precision (fp16)** - A training technique that uses 16-bit floating point arithmetic for most computations to reduce memory usage and increase throughput on compatible hardware.

**Domain-adaptive pre-training** - The practice of continuing MLM training on a domain-specific corpus to adapt a general-purpose language model to specialized text.
