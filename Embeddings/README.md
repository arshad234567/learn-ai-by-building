# Embeddings in Natural Language Processing

This document covers three interconnected concepts in modern NLP: token embeddings versus sentence embeddings, contextualized token embeddings, and dual encoders. Together these topics form the foundation of how neural language models represent and compare text.

---

## Table of Contents

1. [Token Embeddings vs Sentence Embeddings](#1-token-embeddings-vs-sentence-embeddings)
2. [Contextualized Token Embeddings](#2-contextualized-token-embeddings)
3. [Dual Encoders](#3-dual-encoders)
4. [How These Concepts Connect](#4-how-these-concepts-connect)

---

## 1. Token Embeddings vs Sentence Embeddings

### What an Embedding Is

An embedding is a dense, fixed-dimensional vector that represents a piece of text in a continuous numerical space. The goal is to place semantically similar pieces of text close together in this vector space, so that geometric operations on the vectors correspond to meaningful relationships in language.

### Token Embeddings

A token is the smallest unit of text a model operates on. Depending on the tokenizer, a token may be a word, a subword fragment, a character, or a punctuation mark. A token embedding maps each token to a vector.

In the simplest case, a model learns a lookup table called an embedding matrix. Each row corresponds to one token in the vocabulary, and the row values are the learned vector for that token. When the model sees the token "bank", it retrieves the corresponding row and uses that vector as the input representation.

The core limitation of this approach is that each token has exactly one vector regardless of context. The word "bank" gets the same vector whether the surrounding sentence is about rivers or financial institutions. This is called a static or non-contextual embedding.

Static token embedding models include Word2Vec, GloVe, and FastText. These models were influential because they demonstrated that useful semantic relationships could be captured geometrically. The classic example is that the vector arithmetic `king - man + woman` produces a vector close to `queen`. However, the inability to disambiguate meaning by context is a significant constraint for downstream tasks.

### Sentence Embeddings

A sentence embedding represents an entire sentence, paragraph, or document as a single vector. The goal is to capture the overall meaning of a piece of text rather than the meaning of individual tokens.

There are several strategies for producing a sentence embedding from token-level representations:

**Mean pooling** averages the vectors of all tokens in the sentence. This is simple and often effective, but it weights every token equally regardless of importance.

**CLS token pooling** uses the vector of a special classification token, typically written as [CLS], that is prepended to the input. In BERT-style models, this token is designed to aggregate information from the entire sequence through the attention mechanism.

**Max pooling** takes the element-wise maximum across all token vectors, which tends to capture the most prominent features present anywhere in the sequence.

Sentence embeddings are the right tool when you need to compare or retrieve entire pieces of text. Common applications include semantic search, document clustering, duplicate detection, and sentence similarity scoring. The single-vector representation makes it straightforward to compute similarity with a dot product or cosine similarity.

### The Tradeoff

Token embeddings preserve fine-grained, position-specific information. They are the right choice when a downstream task requires understanding individual words or spans within a sequence, such as named entity recognition, span extraction in question answering, or token classification.

Sentence embeddings compress everything into a single vector, which loses positional and token-level detail but makes cross-document comparison efficient. Choosing between them depends on the granularity at which the task operates.

---

## 2. Contextualized Token Embeddings

### The Problem with Static Embeddings

Static embeddings assign one fixed vector per token regardless of where or how it appears. Natural language is deeply ambiguous. The same word can mean different things in different sentences, and the meaning of a word is often determined entirely by the words around it. A model that cannot adjust a token's representation based on its context will make systematic errors on tasks requiring disambiguation.

### The Contextualizing Mechanism

Contextualized token embeddings solve this by producing a different vector for each token depending on the full input sequence. Two occurrences of the same token in different sentences will receive different vectors if their contexts differ.

The dominant mechanism for producing contextual representations is the Transformer architecture, specifically the multi-head self-attention mechanism. Self-attention allows every token in a sequence to directly attend to every other token. For each token, the model computes a weighted sum of the representations of all other tokens, where the weights are determined by how relevant each other token is to the current one. This means the final representation of a token encodes not just the token itself but information about its relationships to everything else in the sequence.

After attention, representations are passed through feed-forward layers, normalized, and then passed through the same structure repeatedly across multiple Transformer layers. In deep models like BERT-large or RoBERTa, this process runs across 24 layers. Each layer refines the representations using information aggregated in the previous layer.

The result is that the vector produced for the token "bank" in "river bank" will differ substantially from the vector produced for "bank" in "bank account", because the attention patterns connecting those tokens to their surroundings will differ.

### Pre-training for Contextual Representations

Contextualized embeddings of the kind used in practice today come from large pre-trained models. Pre-training exposes the model to vast amounts of text with self-supervised objectives that force it to develop rich contextual representations.

The two most influential pre-training objectives are:

**Masked Language Modeling (MLM)** randomly replaces some tokens in the input with a mask token and trains the model to predict the original tokens from context alone. Because predicting a masked word requires understanding its context, the model is forced to learn relationships between tokens.

**Next Sentence Prediction (NSP)** trains the model to predict whether two sentences appear consecutively in the original text. This encourages representations that capture discourse-level relationships, though later research found this objective less critical than MLM.

After pre-training, the model's parameters encode a general-purpose understanding of language. These parameters can then be fine-tuned on a specific downstream task with a relatively small labeled dataset, which is far more efficient than training a task-specific model from scratch.

### Layers and Representation Quality

Different layers of a Transformer tend to capture different kinds of information. Earlier layers tend to capture syntactic properties such as part-of-speech and dependency structure. Later layers tend to capture more semantic and task-relevant information. For many downstream tasks, using the final layer's representations performs best, but for some tasks, averaging representations from multiple layers produces better results.

### Applications

Contextualized token embeddings are used wherever fine-grained, position-specific understanding is required. Named entity recognition uses per-token representations to label each token with an entity type. Span-based question answering identifies the start and end token positions of an answer within a passage. Coreference resolution links tokens that refer to the same real-world entity. Relation extraction identifies which tokens describe entities and which describe the relationship between them.

---

## 3. Dual Encoders

### Motivation

Many practical NLP tasks require comparing a query against a large collection of candidates to find the most relevant ones. Examples include:

- Open-domain question answering, where a question must be matched to a relevant passage in a large corpus
- Semantic search, where a user query must be matched to documents
- Dialogue systems, where an utterance must be matched to a response
- Entity linking, where a mention in text must be matched to an entry in a knowledge base

A naive approach is to concatenate the query and each candidate and run them through a cross-attention model that considers both simultaneously. This produces highly accurate relevance scores because the model can directly compare every token in the query to every token in the candidate. However, this is computationally prohibitive at retrieval scale because the number of forward passes required is the number of candidates, which may be millions or billions.

### The Dual Encoder Architecture

A dual encoder (also called a bi-encoder or two-tower model) addresses this by encoding the query and each candidate independently with separate or shared encoders. Each encoder produces a single dense vector for its input. Relevance is then scored by a simple operation, typically dot product or cosine similarity, between the query vector and the candidate vector.

The two encoders are often identical in architecture and share weights, though they can also be separate models trained with different objectives. The defining property is that the two inputs are never processed jointly. There is no cross-attention between query tokens and candidate tokens.

The key computational advantage is that candidate vectors can be computed offline and stored in an index. At query time, only the query needs to pass through the encoder in real time. The similarity between the query vector and every stored candidate vector can then be computed efficiently using approximate nearest neighbor search, enabling retrieval from very large corpora in milliseconds.

### Training Dual Encoders

The standard training objective for dual encoders is contrastive learning. The model is given batches of positive pairs, for example a question and the passage that contains its answer. For each positive pair in the batch, the other candidates in the batch serve as negatives. The model is trained to assign high similarity to positive pairs and low similarity to negative pairs.

The most widely used loss function is the cross-entropy loss over in-batch negatives. For a query with a known positive candidate, the model computes the similarity between the query vector and all candidate vectors in the batch. The loss pushes the similarity of the positive candidate to be high relative to the similarities of the negatives.

**Hard negatives** are candidates that are superficially similar to the positive but actually irrelevant. Training with hard negatives forces the model to learn fine-grained distinctions and significantly improves retrieval quality compared to using only random in-batch negatives. Hard negatives are often obtained by running a trained retriever on the training queries, collecting high-scoring but incorrect candidates, and including those in subsequent training rounds.

### Limitations

Because the query and candidate are encoded independently, a dual encoder cannot directly compare token-level interactions between the two. It must compress all relevant information into a single vector before comparing. This is a strict constraint on the kinds of relevance signals the model can use.

Cross-encoder models that process query and candidate jointly consistently outperform dual encoders on relevance scoring because they have access to cross-attention. The practical solution in many production systems is a two-stage pipeline: a dual encoder performs fast first-stage retrieval to reduce the candidate set from millions to hundreds, and then a cross-encoder re-ranks those hundreds with full cross-attention. The dual encoder handles scale; the cross-encoder handles accuracy.

### Dense Passage Retrieval

Dense Passage Retrieval (DPR) is a widely cited instantiation of the dual encoder approach for open-domain question answering. It uses two separate BERT encoders, one for questions and one for passages, trained with in-batch negatives and hard negatives mined from an initial BM25-based retrieval pass. DPR demonstrated that dense retrieval with a dual encoder can substantially outperform traditional term-matching retrieval methods like BM25 on several open-domain QA benchmarks.

### Applications

Dual encoders are used in any system that must retrieve from a large candidate set in near real time. This includes web-scale semantic search engines, open-domain question answering pipelines, product recommendation systems that match user queries to catalog items, and dialogue systems that retrieve relevant responses from a large set of candidates.

---

## 4. How These Concepts Connect

These three topics build on each other in a direct progression.

Static token embeddings were the first generation of learned text representations. They demonstrated that geometric structure in vector space corresponds to semantic structure in language, but they were limited by the absence of context sensitivity.

Contextualized token embeddings overcame this limitation by using Transformer self-attention to make each token's representation dependent on its full surrounding context. Pre-trained models like BERT made these rich representations available without requiring large labeled datasets for every task.

Sentence embeddings, obtained by pooling contextualized token representations, provide a single vector per text unit. These vectors can be compared with simple dot products, enabling efficient similarity computation at scale.

Dual encoders operationalize this by training two encoders to produce query and candidate sentence embeddings that are directly comparable. The architecture separates the encoding of the two inputs, which is what makes large-scale retrieval computationally feasible. The quality of the dual encoder depends entirely on the quality of the underlying contextualized representations, which is why pre-trained Transformer models are universally used as the backbone.

Together, these ideas underpin the retrieval components of modern question answering systems, semantic search engines, and retrieval-augmented generation pipelines.

---

## Glossary

**Token** - The basic unit of text processed by a model. May be a word, subword, or character depending on the tokenizer.

**Embedding** - A dense vector representation of text in a continuous numerical space.

**Static embedding** - An embedding where each token has a single fixed vector regardless of context.

**Contextualized embedding** - An embedding where a token's vector is computed from its surrounding context, so the same token can have different vectors in different sentences.

**Self-attention** - A mechanism that computes each token's representation as a weighted combination of all other tokens in the sequence, where weights reflect relevance.

**Pooling** - The operation of aggregating multiple token vectors into a single sentence vector. Common strategies are mean pooling, max pooling, and CLS token pooling.

**Dual encoder** - A model architecture with two independent encoders that produce vectors for a query and a candidate, compared via dot product or cosine similarity.

**Contrastive learning** - A training paradigm that pushes representations of positive pairs closer together and negative pairs further apart in vector space.

**Hard negative** - A training negative that is superficially similar to a positive example, used to train finer-grained distinctions.

**Cross-encoder** - A model that processes query and candidate jointly with cross-attention, producing more accurate but slower relevance scores than a dual encoder.

**Approximate nearest neighbor (ANN) search** - An algorithm for efficiently finding vectors in a large index that are closest to a query vector, without exhaustively comparing against every stored vector.
