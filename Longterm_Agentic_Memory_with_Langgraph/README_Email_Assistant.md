# Email Assistant with LangGraph and Memory

This document covers the design and implementation of an AI-powered email assistant built across three progressive stages. The assistant grows from a stateless triage and response agent into a fully memory-enabled system capable of learning from past decisions and retaining knowledge about contacts and preferences across conversations.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Triage: Classifying Incoming Email](#2-triage-classifying-incoming-email)
3. [The Response Agent and Tools](#3-the-response-agent-and-tools)
4. [Graph Architecture with LangGraph](#4-graph-architecture-with-langgraph)
5. [Semantic Memory](#5-semantic-memory)
6. [Episodic Memory and Few-Shot Learning](#6-episodic-memory-and-few-shot-learning)
7. [How the Three Notebooks Relate](#7-how-the-three-notebooks-relate)
8. [Glossary](#8-glossary)

---

## 1. System Overview

The email assistant is an agentic system that processes incoming emails on behalf of a user. It is structured as two cooperating components: a triage router that decides what to do with an email, and a response agent that takes action when a reply is warranted.

The user is modeled through two data structures. The profile holds static identity information: name, full name, and background context describing the user's role. The prompt instructions hold behavioral rules that govern how the assistant should handle different categories of email. Together, these two structures parameterize every prompt sent to the underlying language model, so the assistant's behavior is entirely shaped by configuration rather than hard-coded decisions.

The system uses a language model for two distinct purposes: structured classification during triage, and open-ended tool-calling reasoning during response. These two tasks require different output formats and different system prompts, so they are driven by two separate model configurations.

---

## 2. Triage: Classifying Incoming Email

### The Classification Problem

Not every email deserves the same response. Some emails are irrelevant and should be silently discarded. Some carry information the user should be aware of but do not require a reply. Others demand direct action from the user. The first task of the assistant is to sort an incoming email into one of these three categories: ignore, notify, or respond.

This classification is driven by configurable triage rules. The rules are expressed in plain natural language and injected into the system prompt at runtime. This means the classification logic is not written in code; it lives in the configuration and can be updated without changing any code.

### Structured Output with a Router Model

Rather than asking the language model to produce free-form text and then parsing that text to extract a classification decision, the assistant uses structured output. A Pydantic model called Router defines the exact schema the model must produce: a reasoning field that contains a chain-of-thought explanation of the classification decision, and a classification field restricted to exactly three string values.

The language model is configured to emit output conforming to this schema using the with_structured_output method. This converts the raw language model into a model that returns typed Python objects. The classification field is declared using Python's Literal type, which restricts it to the values "ignore", "notify", and "respond" at the type level. If the model produces anything outside these values, validation fails immediately rather than propagating a bad decision silently downstream.

The reasoning field is important even though its value is not used to drive routing decisions. Asking the model to produce reasoning before its classification encourages the model to engage in deliberate thinking about the email before committing to a label, which improves classification accuracy. This is the same principle as chain-of-thought prompting.

### The Triage Prompt

The triage system prompt is structured into clearly delineated sections using XML-style tags. This structure is not aesthetic; it helps the language model reliably identify where role information ends, where background context begins, where rules are, and where examples appear. Research and empirical practice with large language models consistently shows that clearly sectioned prompts with consistent delimiters produce more reliable output than flat, unstructured prose.

The prompt is populated at runtime by string formatting. The user's profile fields and triage rules are interpolated into the template before the prompt is sent to the model, making the same prompt template reusable across different users and different rule configurations.

---

## 3. The Response Agent and Tools

### Tools

When the triage router decides that an email requires a response, control passes to the response agent. This agent has access to a set of tools that represent actions it can take in the real world on behalf of the user.

The tools in this system are:

**write_email** takes a recipient address, a subject line, and body content, and sends an email. In a production system this would call an email API. In these notebooks it returns a confirmation string as a placeholder.

**schedule_meeting** takes a list of attendee addresses, a subject, a duration in minutes, and a preferred day, and creates a calendar event. Again, a production implementation would call a calendar API.

**check_calendar_availability** takes a day name and returns a list of free time slots. This allows the agent to check availability before proposing or accepting a meeting time.

**notify** is a special tool present in the baseline notebook that allows the agent to surface important information to the user without sending an email reply. It is the action taken when the triage classification is "notify".

Tools are defined using the tool decorator from LangChain, which wraps a regular Python function and makes its name, description, and argument schema automatically available to the language model. The docstring of the function becomes the tool description, which is what the model reads to decide whether to use the tool in a given situation. Precise, instructive docstrings are therefore important for correct tool selection.

### The ReAct Agent

The response agent is built using the ReAct (Reasoning and Acting) pattern. In this pattern, the language model alternates between producing reasoning about what to do next and selecting a tool to call. After a tool is called, the result is appended to the conversation history, and the model reasons again based on the updated state. This loop continues until the model decides it has completed the task and produces a final text response.

LangGraph provides a prebuilt create_react_agent function that assembles this loop automatically. The agent receives the list of available tools, a prompt factory function, and the language model, and returns a compiled graph that implements the full ReAct loop.

The agent system prompt instructs the model to behave like an executive assistant, to call tools immediately when relevant rather than narrating intent, and to return a final answer directly after tool use without looping unnecessarily. These constraints are important because language models by default tend to verbalize intentions before acting, which is inefficient in an agentic setting where execution speed matters and verbose intermediate reasoning wastes tokens.

### Prompt Factory Function

The agent prompt is not a static string. It is produced by a factory function that receives the current agent state and constructs the full message list to send to the model. The system message is assembled at call time by formatting the system prompt template with the current user profile and agent instructions. This design allows the system prompt to be dynamically updated based on state, which becomes essential when memory is introduced and retrieved context needs to be injected into the prompt.

---

## 4. Graph Architecture with LangGraph

### Why a Graph

The email assistant is implemented as a stateful directed graph using LangGraph. Each node in the graph is a function that receives the current state, performs some operation, and returns either an updated state or a routing decision. Edges connect nodes and define which node runs next.

Using a graph rather than a simple sequential function call chain provides several advantages. Routing decisions can be made dynamically at runtime based on intermediate results, rather than being fixed at definition time. State is explicitly typed and passed through the graph in a controlled way, making the data flow visible and auditable. Adding new nodes or changing routing logic requires only local modifications rather than refactoring a tangled chain of function calls.

### State

The graph state is defined as a TypedDict with two fields. The email_input field holds the incoming email as a dictionary containing the author, recipient, subject, and body of the email thread. The messages field holds the conversation history between the response agent and the language model, accumulated across all tool calls and responses. The messages field uses a special add_messages annotation that instructs LangGraph to append new messages to the existing list rather than replacing it, preserving the full conversation history across graph steps.

### Nodes and Routing

The graph has two nodes: triage_router and response_agent.

The triage_router node reads the email from state, builds and sends the classification prompt, and returns a Command object. A Command in LangGraph combines a routing decision with a state update in a single return value. If the classification is "respond", the command routes to the response_agent node and appends an initial user message to the messages list instructing the agent to draft a reply. If the classification is "ignore" or "notify", the command routes directly to the END node, terminating the graph immediately without invoking the response agent.

The response_agent node is the compiled ReAct agent. It receives the state with the accumulated messages, runs the full tool-calling loop until completion, and appends all intermediate and final messages to the state.

The graph is compiled with a global store object passed in, so that both the triage router and the response agent can access the same shared memory store.

---

## 5. Semantic Memory

### What Semantic Memory Is

Semantic memory is long-term memory that stores general facts, preferences, and knowledge about the world. In the context of the email assistant, semantic memory allows the agent to remember information it has learned across conversations: who a contact is, what their relationship to the user is, what topics have come up in previous interactions, and what preferences or constraints the user has expressed.

Without semantic memory, every invocation of the agent starts from a blank slate. The agent has no recollection of previous emails, previous decisions, or any context established in prior sessions. Semantic memory gives the agent continuity across conversations.

### InMemoryStore and Vector Indexing

Semantic memory is implemented using LangGraph's InMemoryStore. The store is a key-value store organized into namespaces. A namespace is a tuple of strings that identifies a logical partition of the store, for example ("email_assistant", "lance", "collection") partitions the store by application, by user identity, and by data type.

When the store is initialized with an index configuration specifying an embedding model, it becomes a vector store: items written to the store are automatically embedded using the specified model, and the store supports semantic search queries that find stored items by meaning rather than by exact key match.

The embedding models used in these notebooks are sentence-transformer models from Hugging Face, specifically all-MiniLM-L6-v2 and BAAI/bge-small-en. These are compact but capable models that convert text into dense semantic vectors. When new information is written to the store, these models produce the embedding that is stored alongside the text. When a search query is issued, the query is embedded with the same model, and the store returns the stored items whose embeddings are closest to the query embedding.

### Memory Tools

The agent accesses semantic memory through two tools provided by the langmem library: manage_memory and search_memory.

manage_memory allows the agent to write new information into the store during the course of a conversation. When the agent encounters something worth remembering, such as a fact about a contact or a preference the user expressed, it calls this tool with the text to store. The tool handles embedding and persistence automatically.

search_memory allows the agent to query the store for relevant previously stored information before responding to a new message. The agent calls this tool with a natural language query describing what it is looking for, and the store returns the most semantically similar stored memories. The agent can then incorporate that retrieved context into its response.

The namespace for both tools includes a template variable {langgraph_user_id} that is resolved at runtime from the configuration. This means each user has their own isolated namespace in the store. The same store can serve multiple users without any risk of memory bleed between them.

### User Isolation

User isolation is the property that one user's stored memories are never retrieved when handling another user's requests. This is enforced through the namespace structure. Because the user identifier is part of the namespace key, search queries are always scoped to the requesting user's partition of the store. When the email_agent is invoked with a config containing a specific langgraph_user_id, all store operations that session execute against that user's namespace exclusively.

This design is important for multi-user deployment: the same compiled graph and the same store instance can serve many users simultaneously, with memories remaining private to each user.

---

## 6. Episodic Memory and Few-Shot Learning

### What Episodic Memory Is

Episodic memory stores records of specific past events: what happened, when, and what the outcome was. In the context of the email assistant, episodic memory stores past emails alongside the decisions that were made about them: which classification the user confirmed was correct. These stored past decisions serve as examples that can be retrieved and injected into the triage prompt to guide the model's current classification decision.

This technique is called few-shot prompting. Rather than relying purely on the general triage rules defined in the system prompt, the model is also shown a small number of concrete examples drawn from its past experience with this particular user. These examples calibrate the model's behavior toward the user's actual preferences, which may differ in nuance from what the general rules express.

### Storing Episodes

Episodes are stored in the InMemoryStore under a dedicated namespace, for example ("email_assistant", "lance", "examples"). Each episode is a dictionary containing the original email and the label that was assigned to it. Episodes are added to the store using store.put with a randomly generated UUID as the key.

When an episode needs to be retrieved, store.search is called against the examples namespace. The query is a string representation of the current incoming email. Because the store uses semantic vector search, it returns the stored examples that are most semantically similar to the current email, even if the current email does not share exact wording with any stored example. This allows the assistant to retrieve relevant prior cases based on topic and intent rather than requiring exact match.

### Formatting Examples for the Prompt

The retrieved examples are formatted into a structured block of text using a template that represents each example as its subject, sender, recipient, truncated body content, and the triage decision that was made. These formatted examples are injected into the triage system prompt in the few-shot examples section.

The triage system prompt instructs the model to follow the examples more strongly than the general rules. This gives the user a mechanism to correct the assistant's behavior: if the model repeatedly makes a wrong classification decision, the user can store a labeled example demonstrating the correct decision, and future invocations will be guided by that correction. The assistant learns from demonstrated corrections rather than requiring the user to reformulate rules in abstract language.

### Semantic Retrieval of Examples

The key property that makes episodic memory useful is that example retrieval is semantic rather than lexical. When a new email arrives, the query used to search the examples store is a string representation of the full email including subject, sender, and body. The vector similarity search then finds the stored past emails that are most topically and stylistically similar, even if none of them share the exact words used in the current email.

This means that a stored example about a spam email offering to sell software documentation will be retrieved as a relevant precedent when a new email offering to sell API documentation arrives, because the two emails occupy nearby positions in the embedding space despite differing in surface wording. The assistant can generalize from past examples rather than requiring a fresh explicit correction for every superficially novel variation of a recurring situation.

### User-Scoped Episodes

Episodic memory follows the same user isolation pattern as semantic memory. The namespace includes the user identifier, so each user's past examples are stored and retrieved independently. An episode stored for user "Arshad" will not influence the triage decisions made for user "Akram", even though both use the same deployed graph and store.

---

## 7. How the Three Notebooks Relate

The three notebooks represent a staged development of the same email assistant concept, with each stage adding a new capability on top of the previous one.

The first notebook, email_assistant, establishes the foundational architecture: a triage router using structured output to classify emails into three categories, a response agent with tools for writing emails, scheduling meetings, and checking calendar availability, and a LangGraph state machine that connects them. This version is entirely stateless. Each email is handled in isolation with no memory of previous emails or interactions.

The second notebook, Email_Assistant_with_Semantic_Memory, introduces the InMemoryStore with vector indexing and adds the manage_memory and search_memory tools to the response agent's tool set. The agent can now write facts into persistent storage during a conversation and retrieve them in future conversations. The triage router itself remains unchanged: it still classifies each email based solely on the fixed rules and the current email content. But the response agent gains the ability to recall previous context when drafting replies, enabling more coherent and personalized responses across sessions.

The third notebook, Email_Assistant_with_Semantic_and_Episodic_Memory, adds episodic memory on top of semantic memory. The triage router is upgraded to retrieve past labeled examples from the store and inject them into the classification prompt as few-shot demonstrations. The store is now used by both the triage router, for retrieving classification examples, and the response agent, for retrieving general factual memories. The system also demonstrates that user isolation works correctly by running the same email through the agent with two different user identities and observing that the example store for one user does not affect the behavior observed for the other.

The progression from stateless to semantic memory to episodic memory represents a general principle in building production-quality agents: start with explicit rules and deterministic routing, add semantic memory to give the agent continuity and recall, and add episodic memory to give the agent the ability to learn from demonstrated examples and adapt its behavior to the specific preferences of each individual user.

---

## 8. Glossary

**Triage** - The process of sorting incoming items, here emails, into categories by urgency and required action, before any action is taken.

**Structured output** - A technique where a language model is constrained to produce output conforming to a specified data schema, such as a Pydantic model, rather than free-form text.

**Pydantic** - A Python library for defining data schemas using type annotations. Used here to define the Router schema that constrains the model's classification output.

**Literal type** - A Python type annotation that restricts a value to one of a fixed set of named constants. Used here to ensure the classification field can only be "ignore", "notify", or "respond".

**Chain-of-thought** - A prompting technique that asks the model to produce intermediate reasoning steps before arriving at a final answer, improving accuracy on classification and reasoning tasks.

**LangGraph** - A library for building stateful agent workflows as directed graphs, where nodes are functions and edges define routing between them.

**StateGraph** - The LangGraph class used to define a graph. Nodes and edges are added to it before it is compiled into an executable agent.

**Command** - A LangGraph return type that combines a routing decision (which node to go to next) with a state update in a single object.

**ReAct** - Reasoning and Acting, an agent pattern where the model alternates between reasoning about what to do and calling a tool to do it, continuing until the task is complete.

**Tool** - A Python function wrapped with metadata (name, description, argument schema) that a language model can invoke as part of an agentic workflow.

**InMemoryStore** - A LangGraph key-value store that lives in process memory and, when configured with an embedding model, supports semantic similarity search over stored items.

**Namespace** - A tuple of strings used to partition a store into logical sections. Items written to one namespace are only retrievable from that namespace.

**Semantic memory** - Long-term memory that stores general facts and knowledge about contacts, preferences, and context, retrievable by meaning using vector similarity search.

**Episodic memory** - Long-term memory that stores records of specific past events (emails and their classifications) used as examples to guide future decisions.

**Few-shot prompting** - A technique where a small number of labeled examples are included in the prompt to demonstrate the desired behavior to the model before it processes a new input.

**Vector similarity search** - A search method that finds stored items whose embedding vectors are closest to a query embedding vector, returning results by semantic meaning rather than keyword match.

**User isolation** - The property that one user's stored memories cannot be retrieved when handling another user's requests, enforced through namespace partitioning in the store.

**langmem** - A library that provides pre-built memory management tools (manage_memory and search_memory) for use with LangGraph agents and stores.
