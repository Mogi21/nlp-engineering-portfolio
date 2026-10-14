# Applied NLP Pipelines

This repository contains a professional collection of Natural Language Processing (NLP) projects, showcasing end-to-end implementations ranging from classical text analytics to state-of-the-art Generative AI and Retrieval-Augmented Generation (RAG) architectures. 

---

## 📂 Repository Structure & Projects

### 1. Classical NLP & Text Analytics Pipeline
* **`01_classical_nlp_pipeline.ipynb`**
  * **Dataset:** BBC Politics & Business Corpus.
  * **Objective:** Standardize text data and extract meaningful entities and multi-word phrases using classical NLP heuristics.
  * **Key Features:**
    * **Lexical Processing:** Tokenization, case-insensitive stopword filtering, and a comparative analysis of Lemmatization (WordNet) versus Stemming (Porter).
    * **Syntactic Analysis:** Part-of-Speech (POS) tagging and shallow parsing (Chunking) to extract Named Entities (Persons, Organizations, Geo-Political Entities).
    * **N-Gram Extraction:** Trigram frequency analysis to surface common contextual phrases.
    * **Regex Engine:** Custom context-aware regular expression extraction to isolate target keywords and their immediate surrounding context without relying on deep dependency parsing.

### 2. Retrieval-Augmented Generation (RAG) QA System
* **`02_rag_qa_system.ipynb`**
  * **Dataset:** Natural Questions (NQ) benchmark and Wikipedia Sparse Index.
  * **Objective:** Build a robust Open-Domain Question Answering system using the Retrieve-and-Read paradigm to handle knowledge-intensive tasks.
  * **Key Features:**
    * **First-Stage Retrieval:** BM25 sparse retrieval leveraging PyTerrier.
    * **Neural Re-Ranking:** MonoT5 integration to optimize the ranking of retrieved contexts based on semantic relevance.
    * **Query Expansion:** Pseudo-Relevance Feedback using RM3 to augment initial queries.
    * **Generative Reading (Zero-Shot):** Integration of sequence-to-sequence models (Flan-T5) and Fusion-in-Decoder (FiD) architectures to synthesize final answers from the retrieved passages.
    * **Pipeline Benchmarking:** Automated evaluation frameworks using Exact Match (EM) and F1 scores to measure the performance delta between isolated retrieval and end-to-end generation.

### 3. LLM Bias & Ethics Audit (Named Entity Recognition)
* **`03_ner_bias_audit.ipynb`** *(Pending Upload)*
  * **Dataset:** CoNLL 2003 & Custom Demographic Test Sets.
  * **Objective:** Audit a pre-trained BERT classification model for demographic and gender biases.
  * **Key Features:** Transformer fine-tuning evaluation, confusion matrix analytics, and disparate impact analysis across synthesized ethnic and gender-specific entity subsets.

---

## 🛠️ Tech Stack
* **Core Frameworks:** Hugging Face `transformers`, `datasets`, PyTorch
* **Information Retrieval:** PyTerrier (`pt`, `pyterrier_rag`, BM25, MonoT5)
* **Generative LLMs:** `google/flan-t5-base`, `terrierteam/t5fid_base_nq`
* **Classical NLP:** NLTK (`wordnet`, `averaged_perceptron_tagger`, `maxent_ne_chunker`), Regular Expressions (`re`)
* **Data Processing & Evaluation:** Pandas, NumPy, Scikit-Learn (`confusion_matrix`, `f1_score`)
