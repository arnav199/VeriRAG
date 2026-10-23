# VeriRAG Learning Guide

This guide connects the code to the main ideas behind retrieval-augmented generation (RAG). The sample examples use a tiny library policy so you can verify each step by reading the passage yourself.

## 1. Chunks and metadata

A PDF page can contain more text than is useful to send to a language model at once. VeriRAG breaks page text into smaller passages called **chunks**. Each chunk stores a document ID, filename, page number, and chunk ID. For example, “Documents are kept until the researcher deletes them” may be one chunk tied to page 1.

Chunk size is a trade-off. Very small chunks can lose context; very large chunks can include unrelated material. Overlap repeats a small amount of text at boundaries so a fact split between two chunks is less likely to be lost. This prototype uses character counts for clarity; production systems often use tokenizer-aware chunking.

## 2. Embeddings

An **embedding** is a list of numbers that represents text in a way that places related meanings near each other. A model maps “When are files removed?” and “kept until the researcher deletes them” to vectors that should be closer than unrelated passages.

VeriRAG uses a pretrained Sentence Transformers model locally. It does not train the model. The first run downloads model files; after that, encoding can run on the local CPU.

## 3. Cosine similarity and vector search

**Cosine similarity** compares the direction of two vectors. A score near 1 means the vectors point in a similar direction, while a score near 0 means they are less aligned. VeriRAG normalizes embeddings and uses FAISS inner-product search, which then corresponds to cosine similarity.

FAISS is a library for quickly searching vectors. It returns nearby chunk vectors for a question. Those scores help find relevant passages; they do not prove that a passage supports a claim. Similar wording can be misleading, and a contradiction can still be semantically similar.

## 4. Retrieval-augmented generation

**RAG** combines search with text generation:

1. Embed the question.
2. Retrieve nearby document chunks.
3. Give the question and passages to a language model.
4. Ask it to answer only from those passages and cite their chunk IDs.

For example, a question about file retention retrieves the retention paragraph and passes it with the question. The generator can still make mistakes, so the returned evidence is shown beside the answer.

## 5. Hallucination and faithfulness

A **hallucination** is a generated statement that is false or unsupported. **Faithfulness** describes whether an answer stays grounded in the supplied evidence. A statement can be true in the outside world but unsupported by the uploaded files; VeriRAG must not treat those as the same thing.

If the files do not mention a cloud region, the correct document-grounded response is that the documents do not say. Missing evidence is not proof that a claim is false.

## 6. Claim extraction and evidence verification

Claim extraction splits a paragraph into concise factual statements. For example, “Files stay local, and users can delete them” becomes two claims. Splitting helps identify which parts have evidence and which need review.

The verifier compares each claim with retrieved passages. VeriRAG supports `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT_EVIDENCE`. Contradiction is separate from lack of support: a passage saying “kept until deletion” does not address encryption, while a passage saying “deleted after thirty days” conflicts with a claim that files are kept indefinitely.

The similarity baseline uses rough cosine thresholds to flag topical closeness. It is a teaching baseline, not a truth detector. The LLM verifier reads the claim and evidence and returns a structured assessment, but it can still misread the text. Evidence IDs are checked against the actual retrieved chunks so the API cannot accept a made-up source reference.

## 7. Evaluation metrics

Metrics need human labels or references. For binary classification of a label such as `SUPPORTED`:

- **Precision** = correctly predicted supported claims / all claims predicted supported.
- **Recall** = correctly predicted supported claims / all claims that are actually supported.
- **F1** = the harmonic mean of precision and recall.

The unsupported-claim rate counts `UNSUPPORTED` and `CONTRADICTED` assessments among all claims. It describes this dataset and verifier, not all hallucinations. Answer token F1 measures word overlap with a reference answer, not meaning. Retrieval precision/recall and citation support rate require manually annotated relevant chunk IDs.

Keep question sets and annotations fixed when comparing ordinary RAG with verification-enhanced RAG. Report latency and failures too. A small dataset is useful for debugging and learning, but it cannot establish broad effectiveness or research novelty.

