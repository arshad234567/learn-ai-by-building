# Serving LLMs Efficiently with vLLM

## Overview

Large language models are computationally expensive to serve. Making them smaller through compression is only half the problem. The other half is serving them efficiently to many users at once, without wasting GPU resources or running out of memory. This document explains the three core inference optimization techniques that power vLLM: Continuous Batching, PagedAttention, and Prefix Caching.

---

## Background: Why Inference Optimization Matters

LLM text generation is inherently iterative. Every single token requires a complete forward pass through the model, which means pulling all of the model weights from High Bandwidth Memory (HBM) into the GPU's compute units. Serving one request at a time is therefore deeply wasteful. The tensor cores end up spending most of their time waiting for data rather than doing computation.

Throughput, defined as the number of tokens or requests processed per second across all users, is the key metric that inference optimization aims to improve. The three techniques below address two distinct bottlenecks: GPU compute utilization and GPU memory management.

---

## Technique 1: Continuous Batching

### The Problem with Serving One Request at a Time

When a GPU serves a single request, it reads the entire model from memory for each token generated. The compute required for a single token is tiny relative to the cost of loading the model weights. This means the GPU is dramatically underutilized.

### Why Batching Helps

Batching means processing multiple requests together. Instead of loading model weights for one user, the system loads them once and uses them for many users simultaneously. The memory bandwidth cost stays the same, but far more useful work is done per read.

### Static Batching and Its Limitations

The simplest form of batching is static batching. A fixed group of requests is collected, processed together, and the entire batch waits until every request has finished before the next batch begins. This works well for models like BERT or YOLO, where input and output sizes are predictable. A classification model that takes one image and returns one label has a fixed runtime, so batching ten images together means they all finish at roughly the same time and the GPU stays busy.

LLMs break this assumption entirely. Consider four requests in the same batch. One user asks a short question and gets a five-token answer, finishing at step T5. Another user asks for a two-thousand word essay and finishes at step T8. With static batching, the GPU slot occupied by the short request sits completely idle from T5 to T8, waiting for the long request to complete. The short request is stuck, and the GPU is wasting capacity on an already-finished job.

### How Continuous Batching Solves This

Continuous batching operates at the token level rather than the batch level. Instead of waiting for the entire batch to finish, the scheduler monitors individual requests. The moment any request finishes generating its last token, a new request immediately takes its slot in the batch.

For example, when a short request finishes at T5, a new incoming request is inserted at T5 rather than waiting until T8. The batch is never idle. GPU slots are always occupied by active, useful work. Visually, continuous batching fills GPU time completely, while static batching leaves visible gaps wherever a short request finishes before a long one.

---

## Technique 2: PagedAttention

### The KV Cache Problem

Beyond compute utilization, GPU memory is the second major bottleneck for concurrent request serving. The largest consumer of that memory is the KV cache.

During generation, every active request maintains a Key-Value cache that stores the keys and values for every token generated so far. This cache is essential for attention computation and grows by one entry with every new token. The more concurrent users, the more KV cache memory is required.

The KV cache is particularly difficult to manage for two reasons. First, it grows and shrinks dynamically over the lifetime of a request. Second, you cannot know in advance how long a request will be. Some requests generate five tokens; others generate two thousand.

### How Earlier Systems Handled Memory: Fragmentation

Earlier inference systems handled the KV cache by pre-allocating a contiguous memory block sized to the maximum possible output length for each request. For example, a system might reserve 2048 slots for every request. If a request actually uses 50 tokens, the remaining 1998 slots sit allocated but empty for the entire lifetime of the request. This is called internal fragmentation: wasted space inside an allocation.

There is also external fragmentation. The gaps between pre-allocated blocks for different requests may be physically free but too small to fit another request's pre-allocated chunk, so they go unused as well.

A third form of waste comes from over-reservation. Even the slots a request will eventually use sit reserved and empty for most of its lifetime, blocking other requests from using that memory in the meantime.

The research behind vLLM found that only 20 to 40 percent of KV cache memory in earlier systems was actually used to store real tokens. The remaining 60 to 80 percent was lost to fragmentation and over-reservation. The GPU might have abundant memory in theory, but in practice most of it was locked up and unavailable. This limits how many requests fit in a batch, which directly limits throughput.

### How PagedAttention Works

PagedAttention is the core innovation introduced by vLLM. The idea is borrowed directly from virtual memory and paging in operating systems. When a computer runs a program, the operating system does not reserve one large contiguous chunk of RAM. Instead, it splits memory into small fixed-size pages and scatters them wherever there is room, using a page table to keep track of where everything is. PagedAttention applies exactly this technique to the KV cache.

Instead of storing the KV cache as one large contiguous block, PagedAttention breaks it into fixed-size blocks, also called pages. Each block holds the keys and values for a small fixed number of tokens. The system maintains a block table that maps each request's token positions to the physical blocks holding their keys and values.

Walking through an example step by step makes this concrete.

A prompt arrives. The system grabs one free block from the memory pool, for example physical block 3, and stores the KV cache for the prompt tokens there. The block table records that block 3 is in use with its current fill count.

The model generates the next token. If block 3 still has an empty slot, the new token's KV entry is placed there and the block table is updated. No new allocation is needed yet.

The model generates more tokens until block 3 is full. At that point, the system grabs another free block, for example block 6, and stores the next token there. Critically, block 6 does not need to be physically adjacent to block 3 in memory. They can be in completely different locations. The block table stitches them together logically.

To compute attention for a new token, the model needs to attend to all previous tokens. The system reads the block table, fetches block 3 from GPU memory, computes attention against those tokens, then does the same for block 6. The model successfully attends to all previous tokens even though they are stored in non-contiguous physical memory, because the block table makes the stitching transparent.

Multiple concurrent requests each have their own entries in the block table, with their blocks scattered wherever free space exists. There is no pre-allocation, no reserved empty slots, and no fragmentation. Each request uses exactly the memory it needs at the moment it needs it, and nothing more. This is how vLLM fits significantly more concurrent requests into the same GPU compared to earlier systems.

---

## Technique 3: Prefix Caching

### Repeated Computation Across Requests

Many production LLM deployments share the same prefix across many requests. A system prompt, a set of few-shot examples, or a large RAG context chunk may be identical across every user hitting the same deployment. Without any optimization, every new request recomputes the KV cache for that shared prefix from scratch, even though the computation is identical every time.

Multi-turn conversations create the same problem from a different angle. When a user sends a second message, the full context includes the entire first round of conversation plus the new message. The first round's KV cache was already computed, but without prefix caching, the system recomputes it again as part of processing the second turn.

### How Prefix Caching Solves This

Prefix Caching detects when a new request begins with tokens that have already been processed. Rather than recomputing the KV cache for those tokens, it retrieves the previously computed cache from memory.

For shared system prompts across many users, the prompt's KV cache is computed once on the first request and then reused for every subsequent request. The same applies to shared few-shot examples or shared RAG context documents. For multi-turn conversations, only the new tokens in each round require computation. The model does new work only on new content.

The performance impact is substantial. Benchmarks show that as the cache hit rate increases, throughput increases proportionally. At a 75 percent cache hit rate, throughput is approximately four times higher than with no caching. That factor of four represents computation the system simply does not perform.

---

## How These Techniques Come Together in vLLM

vLLM is the open-source inference engine that combines all three of these techniques into a single serving framework. Continuous batching keeps the GPU fully occupied. PagedAttention eliminates memory fragmentation and maximizes the number of concurrent requests that fit in memory. Prefix caching eliminates redundant computation for shared content. Together they push both throughput and memory efficiency far beyond what earlier systems could achieve.

vLLM supports a broad range of models including Llama, Qwen, DeepSeek, Gemma, Mistral, and Granite, and runs across a wide range of hardware accelerators including NVIDIA GPUs, AMD Instinct, Intel Gaudi, Google TPUs, and AWS Neuron. It can be deployed across edge environments, private cloud, and public cloud from a single unified platform.

By January 2025, vLLM was seeing over 100,000 daily installs, with usage growing tenfold over 2024, reflecting broad adoption across the production AI community.

---

## Summary of Techniques

| Technique | Bottleneck Addressed | Core Mechanism |
|---|---|---|
| Continuous Batching | GPU compute utilization | Replace finished requests immediately at the token level rather than waiting for the full batch |
| PagedAttention | GPU memory fragmentation | Break KV cache into fixed-size non-contiguous pages managed by a block table, analogous to OS virtual memory |
| Prefix Caching | Redundant KV computation | Detect shared token prefixes across requests and reuse previously computed KV cache blocks |

---

## Key Definitions

**KV Cache**: The stored keys and values for all previously generated tokens in a request. Required for attention computation during generation. Grows by one entry per token.

**Internal Fragmentation**: Wasted memory inside an allocation. In the context of KV caches, this is the empty reserved slots in a pre-allocated block that the request never actually fills.

**External Fragmentation**: Wasted memory between allocations. Free gaps that are too small to fit any new request's pre-allocated block.

**Throughput**: The number of tokens or requests a serving system processes per second across all concurrent users.

**HBM (High Bandwidth Memory)**: The GPU memory from which model weights are loaded during each forward pass. Memory bandwidth to HBM is often the primary bottleneck in LLM inference.

**Block Table**: In PagedAttention, the data structure that maps each request's logical token positions to the physical memory blocks holding their KV cache entries.
