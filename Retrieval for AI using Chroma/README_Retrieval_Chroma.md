# Retrieval for AI using Chroma

This document explains the concepts behind building and improving a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB as the vector store. It covers four progressive stages: basic embeddings-based retrieval, diagnosing why vector search fails, expanding queries to improve recall, and re-ranking retrieved results for precision.

---

## Table of Contents

1. [Overview: Retrieval-Augmented Generation](#1-overview-retrieval-augmented-generation)
2. [Embeddings-Based Retrieval](#2-embeddings-based-retrieval)
3. [When Vector Search Fails](#3-when-vector-search-fails)
4. [Query Expansion](#4-query-expansion)
5. [Re-ranking with Cross-Encoders](#5-re-ranking-with-cross-encoders)
6. [How the Four Stages Fit Together](#6-how-the-four-stages-fit-together)
7. [Glossary](#7-glossary)

---

## 1. Overview: Retrieval-Augmented Generation

Retrieval-Augmented Generation is an architecture that pairs a large language model with an external knowledge source. Instead of relying solely on what the model learned during pre-training, the system retrieves relevant text from a document collection at query time and supplies that text to the model as context before it generates an answer.

This matters because language models have two structural limitations: their knowledge is frozen at the point their training data was collected, and they cannot access private or specialized documents that were never part of their training set. RAG addresses both problems by giving the model access to a searchable, up-to-date, and domain-specific corpus at inference time.

A RAG pipeline generally consists of four stages:

1. **Ingestion** - source documents are split into smaller chunks and converted into vector embeddings, which are stored in a vector database.
2. **Retrieval** - an incoming query is embedded using the same embedding model, and the vector database returns the chunks whose embeddings are closest to the query embedding.
3. **Augmentation** - the retrieved chunks are inserted into a prompt alongside the original query.
4. **Generation** - a language model produces an answer grounded in the retrieved context rather than relying purely on its internal parameters.

ChromaDB serves as the vector database in this pipeline. It stores document chunks alongside their embeddings and exposes a query interface that performs nearest-neighbor search to find the chunks most semantically similar to a given query.

---

## 2. Embeddings-Based Retrieval

### Chunking the Source Document

Before any retrieval can happen, the source document must be broken into smaller pieces called chunks. A whole PDF or report is too large and too topically diverse to embed as a single vector; a single embedding cannot meaningfully represent dozens of distinct ideas at once. Chunking solves this by splitting the document into smaller, more topically coherent units, each of which gets its own embedding.

The chunking process used in this stage happens in two passes:

**Character-based splitting** divides the raw extracted text using a hierarchy of separators, typically attempting to split on paragraph breaks first, then line breaks, then sentence boundaries, and finally on individual spaces if no larger natural boundary is available. This produces chunks that respect the natural structure of the writing as much as possible while staying under a target character length.

**Token-based splitting** takes the character-based chunks and further divides them according to the token count expected by the embedding model, since the embedding model has a maximum sequence length measured in tokens rather than characters. This second pass ensures no chunk exceeds the model's input limit, which would otherwise cause silent truncation and loss of information.

Splitting in two passes, rather than just token-splitting from the start, preserves more semantic coherence in the initial split before the final pass enforces the hard token limit.

### Generating Embeddings

Each chunk is converted into a dense vector using a sentence embedding model. The embedding model maps semantically similar pieces of text to nearby points in vector space, so that two chunks discussing the same topic in different words still produce vectors that are close together.

These vectors, along with the original chunk text and an identifier, are added to a Chroma collection. A collection in Chroma functions like a table: it stores documents, their embeddings, and any associated metadata together.

### Querying the Collection

When a user submits a query, the same embedding model converts the query text into a vector. Chroma then performs a nearest-neighbor search, comparing the query vector against every stored document vector using a distance metric such as cosine similarity, and returns the documents whose vectors are closest.

### Generating the Final Answer

The retrieved chunks are concatenated and inserted into a prompt that instructs the language model to answer strictly using the provided context, and to explicitly state when the answer cannot be found in that context. This constraint is important: it reduces the likelihood that the model fabricates information not actually present in the retrieved chunks, a failure mode known as hallucination. The prompt, including the retrieved context, is then sent to the language model, which produces the final grounded answer.

---

## 3. When Vector Search Fails

Embeddings-based retrieval is not infallible. Because retrieval depends on the geometric proximity of the query vector to document vectors, queries that are phrased differently from how the relevant information is written in the source documents can fail to retrieve the correct chunks, even when the answer is present somewhere in the corpus.

### Visualizing the Embedding Space

To understand retrieval behavior, the high-dimensional embeddings produced by the embedding model can be projected down to two dimensions using a dimensionality reduction technique such as UMAP (Uniform Manifold Approximation and Projection). UMAP attempts to preserve the local neighborhood structure of the high-dimensional space when flattening it to two dimensions, so that points which were close together originally remain relatively close together in the projection.

By projecting both the full set of document chunk embeddings and a given query's embedding into the same two-dimensional space, it becomes possible to visually inspect where a query lands relative to the documents and which documents were actually retrieved as its nearest neighbors.

### Diagnosing Retrieval Failure

This visualization reveals several patterns that explain why retrieval sometimes underperforms:

**Good retrieval** appears as a query point landing inside or very near a dense cluster of relevant document chunks, with the retrieved chunks (highlighted distinctly) being the closest points to the query.

**Vocabulary mismatch** occurs when a query uses different terminology than the source documents, causing the query embedding to land in a region of the vector space that does not correspond to the chunks that actually contain the answer. Even though the answer exists in the corpus, the surface-level wording difference is large enough that the embedding model does not place the query close to the relevant content.

**Overly broad or vague queries** can land in a position roughly equidistant from several unrelated clusters, causing the nearest neighbors returned to be a mix of marginally relevant chunks from different topics rather than a single coherent and accurate set of results.

**Queries about content not present in the corpus** will still retrieve the nearest chunks by distance, because vector search always returns the k nearest neighbors regardless of whether they are actually relevant. This is a structural property of nearest-neighbor search: it has no built-in mechanism for recognizing that none of the available documents actually answer the query, so it will confidently return its best (but possibly poor) matches.

These diagnostic visualizations motivate the two techniques covered in the remaining sections. Query expansion addresses vocabulary mismatch by reformulating the query to better align with how the answer is likely phrased in the documents. Re-ranking addresses cases where the initial nearest-neighbor retrieval returns a mix of relevant and irrelevant chunks, by applying a more accurate but more expensive scoring model to reorder the candidates.

---

## 4. Query Expansion

Query expansion addresses the vocabulary mismatch problem by transforming or supplementing the original query before it is used for retrieval. Two distinct expansion strategies are commonly used.

### Expansion via Hypothetical Answer Generation (HyDE)

In this approach, a language model is prompted to generate a plausible, detailed hypothetical answer to the original query, without any requirement that the answer be factually accurate. The system prompt explicitly tells the model that the generated answer does not need to be correct; it only needs to read like a document that could plausibly contain the real answer.

This hypothetical answer is then concatenated with the original query, and the combined text is embedded and used as the retrieval query.

The underlying intuition is that a hypothetical answer, even if factually wrong, is likely to use vocabulary, phrasing, and structure similar to the real answer found in the corpus, because both are describing the same underlying topic in the register of a factual document rather than a question. By embedding a statement-style passage rather than a question, the resulting query vector lands closer in the embedding space to the chunks of the source document that actually discuss that topic, because document chunks themselves are statements rather than questions. This technique is referred to as HyDE, short for Hypothetical Document Embeddings.

### Expansion via Multiple Related Queries

In this approach, instead of generating a single hypothetical answer, a language model is prompted to generate several distinct but related questions that approach the original query from different angles. The system prompt instructs the model to produce short, complete questions that together cover different aspects of the original topic.

Each of the generated questions, along with the original query, is used to independently query the vector collection. This produces several separate result sets, one per query variant. The results across all variants are then combined and deduplicated, since the same chunk may be retrieved by more than one of the query variants.

The rationale behind this technique is that a single phrasing of a question may not align well with how the corpus expresses the relevant information, but a set of differently phrased questions covering different facets of the same underlying information need increases the chance that at least one phrasing aligns closely enough with the document vocabulary to retrieve the correct chunks. This also improves recall when the original question is broad and the true answer is actually composed of information spread across multiple distinct sections of the corpus, since different sub-questions can pull in different relevant chunks that a single query would have missed.

### Visualizing the Effect of Expansion

The effect of query expansion can be observed using the same UMAP projection technique described in the previous section. By plotting the original query, the expanded query or queries, and the resulting retrieved chunks together in the same two-dimensional space, it becomes visible whether the expanded queries land closer to the relevant document cluster than the original query did, and whether they pull in a wider or more accurate set of retrieved chunks.

---

## 5. Re-ranking with Cross-Encoders

### Why Re-ranking Is Needed

Vector search using a bi-encoder (also called a dual encoder), where the query and documents are embedded independently and compared by distance, is fast and scales well to large corpora, but it has a structural accuracy ceiling. Because the query and each document are encoded completely independently of one another, the model never gets to directly compare specific tokens in the query against specific tokens in the document. All relevance information must be compressed into a single fixed-size vector for each side before any comparison happens.

A cross-encoder removes this constraint by taking the query and a candidate document together as a single combined input, and computing a single relevance score using full attention between every token in the query and every token in the document. This joint processing allows the model to detect much finer-grained relevance signals that a bi-encoder's compressed vectors cannot capture, which makes cross-encoders substantially more accurate at judging relevance.

The tradeoff is that a cross-encoder must be run once per query-document pair, making it computationally infeasible to use directly against an entire large corpus. The standard solution is a two-stage pipeline: use the fast bi-encoder-based vector search to retrieve a small initial candidate set (for example, the top 10), and then apply the more accurate but slower cross-encoder only to that small candidate set to re-rank them by true relevance.

### How Re-ranking Works in Practice

After the initial nearest-neighbor retrieval returns a set of candidate document chunks, each candidate is paired with the original query to form a query-document pair. The cross-encoder model processes each pair jointly and outputs a single relevance score per pair.

These scores are not bounded probabilities; they are raw relevance scores where higher values indicate stronger relevance and the scores can be negative for clearly irrelevant pairs. Sorting the candidates by this score, from highest to lowest, produces a new ranking that more accurately reflects true relevance to the query than the original distance-based ranking from the vector search step.

### Combining Re-ranking with Query Expansion

Re-ranking and query expansion are complementary rather than competing techniques, and combining them produces a stronger pipeline than either alone. The expanded set of related queries is used to retrieve a broader and more diverse candidate pool from the vector store, increasing recall. The retrieved candidates from all query variants are deduplicated into a single pool. Then the cross-encoder re-ranks this entire deduplicated pool against the original query, ensuring that the final ordering reflects true relevance to what the user actually asked, regardless of which query variant happened to retrieve a given chunk. This combined approach improves both recall, by casting a wider net during retrieval, and precision, by applying an accurate relevance judgment during re-ranking.

---

## 6. How the Four Stages Fit Together

These four notebooks represent a natural progression in building a production-quality retrieval pipeline.

**Embeddings-based retrieval** establishes the baseline pipeline: chunk the source document, embed the chunks, store them in Chroma, embed an incoming query, retrieve the nearest chunks, and generate an answer grounded in that context.

**When vector search fails** introduces a diagnostic methodology, using dimensionality reduction to visualize the embedding space, that reveals why the baseline pipeline sometimes returns poor or irrelevant results. This diagnostic step motivates the need for the two improvement techniques that follow.

**Query expansion** improves recall, the ability to retrieve all the relevant information that exists in the corpus, by reformulating the query through either a hypothetical answer or a set of related sub-questions, addressing the vocabulary mismatch problem identified in the diagnostic stage.

**Re-ranking with cross-encoders** improves precision, the ability to correctly order retrieved results by true relevance, by applying a more accurate but more computationally expensive model to a small candidate set after the fast initial retrieval step.

In a production system, these techniques are typically combined into a single pipeline: expand the query to retrieve a broad, diverse, deduplicated candidate set, then re-rank that candidate set with a cross-encoder, and finally pass only the top-ranked chunks to the language model for answer generation. This combination directly targets the two distinct failure modes that vector search alone is prone to: missing relevant content due to vocabulary mismatch, and surfacing irrelevant content due to the limited discriminative power of distance-based comparison.

---

## 7. Glossary

**Retrieval-Augmented Generation (RAG)** - An architecture that retrieves relevant text from an external knowledge source and supplies it to a language model as context before generation.

**Chunking** - The process of splitting a source document into smaller, more topically coherent pieces, each of which is embedded separately.

**Embedding** - A dense vector representation of text, positioned in vector space such that semantically similar text produces nearby vectors.

**Vector database** - A database, such as ChromaDB, designed to store embeddings alongside associated text and metadata, and to perform efficient nearest-neighbor search over those embeddings.

**Nearest-neighbor search** - The process of finding the stored vectors closest to a given query vector according to a distance metric such as cosine similarity.

**Bi-encoder (dual encoder)** - A retrieval architecture where the query and documents are embedded independently, allowing fast comparison via vector distance but no direct token-level comparison.

**Cross-encoder** - A model that takes a query and a document together as a single input and computes a relevance score using joint attention across both, producing more accurate but more computationally expensive relevance judgments than a bi-encoder.

**UMAP** - Uniform Manifold Approximation and Projection, a dimensionality reduction technique used to project high-dimensional embeddings into two dimensions while preserving local neighborhood structure, enabling visualization.

**Vocabulary mismatch** - A failure mode in vector search where a query is phrased using different terminology than the relevant content in the source documents, causing the query embedding to land far from the chunks that actually answer it.

**Query expansion** - A family of techniques that reformulate or supplement the original query before retrieval, in order to improve the chance of matching relevant document vocabulary.

**HyDE (Hypothetical Document Embeddings)** - A query expansion technique that generates a hypothetical, not-necessarily-correct answer to the query and uses that answer's text, combined with the original query, as the retrieval input.

**Re-ranking** - The process of reordering an initial set of retrieved candidates using a more accurate (and typically more expensive) relevance model, commonly a cross-encoder.

**Recall** - The proportion of all truly relevant information in the corpus that the retrieval process successfully surfaces.

**Precision** - The proportion of retrieved results that are actually relevant to the query.

**Hallucination** - A failure mode in language model generation where the model produces information that is not supported by the provided context or training data.
