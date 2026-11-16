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
2. Retrieve nearby document chunks for answer generation.
3. Give the question and those passages to a language model.
4. Ask it to answer only from those passages and cite their chunk IDs.
5. After claim extraction, separately embed each claim and retrieve claim-specific passages for verification.

For example, a question about file retention retrieves the retention paragraph and passes it with the question. The generator can still make mistakes, so the returned evidence is shown beside the answer.

## 5. Hallucination and faithfulness

A **hallucination** is a generated statement that is false or unsupported. **Faithfulness** describes whether an answer stays grounded in the supplied evidence. A statement can be true in the outside world but unsupported by the uploaded files; VeriRAG must not treat those as the same thing.

If the files do not mention a cloud region, the correct document-grounded response is that the documents do not say. Missing evidence is not proof that a claim is false.

## 6. Claim extraction and claim-specific evidence verification

Claim extraction splits a paragraph into concise factual statements. For example, “Files stay local, and users can delete them” becomes two claims. Splitting helps identify which parts have evidence and which need review.

VeriRAG does a new retrieval for every extracted claim. For example, if an answer says “Cats are mammals. The Pacific is an ocean,” it embeds both claims separately. FAISS can then return the cat passage for the first claim and the Pacific passage for the second. This is different from reusing the question's top passages for every claim. Each claim result includes the candidate chunks, their IDs, filenames, page numbers, document IDs, and similarity ranking scores. The answer-level retrieval remains a separate list used to compose the original response.

The LLM verifier compares a claim only with that claim's retrieved passages. It must return one of `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`, the evidence chunk IDs it assessed, and a non-empty explanation. The backend checks every ID against the claim's candidate set. Missing fields, malformed output, provider failures, and out-of-set IDs cause an API error instead of a guessed label.

Contradiction is separate from lack of support: a passage saying “kept until deletion” does not address encryption, while a passage saying “deleted after thirty days” conflicts with a claim that files are kept indefinitely. `INSUFFICIENT_EVIDENCE` is appropriate when nothing was retrieved or the retrieved passages do not let the verifier decide. `UNSUPPORTED` is a verifier judgment that the retrieved evidence does not substantiate the claim, and depends on retrieval coverage and verifier quality.

The similarity baseline also receives claim-specific retrieval, then applies rough cosine thresholds to flag topical closeness. It is a comparison method, not a truth detector: similarity does not establish factual entailment and its scores are not calibrated truth probabilities. The LLM verifier can still misread relevant text or be given incomplete evidence. Claim-specific retrieval improves relevance of the verifier's input as a design choice, but is not itself evidence of improved answer reliability.

## 7. Evaluation metrics

Metrics need human labels or references. For binary classification of a label such as `SUPPORTED`:

- **Precision** = correctly predicted supported claims / all claims predicted supported.
- **Recall** = correctly predicted supported claims / all claims that are actually supported.
- **F1** = the harmonic mean of precision and recall.

The unsupported-claim rate counts `UNSUPPORTED` and `CONTRADICTED` assessments among all claims. It describes this dataset and verifier, not all hallucinations. Answer token F1 measures word overlap with a reference answer, not meaning. Retrieval precision/recall and citation support rate require manually annotated relevant chunk IDs.

Keep question sets and annotations fixed when comparing ordinary RAG with verification-enhanced RAG. Report latency and failures too. A small dataset is useful for debugging and learning, but it cannot establish broad effectiveness or research novelty.

## 8. Try the claim-specific flow

From the repository root, start the API with `python -m uvicorn backend.app.main:app --reload`, then start the browser app in a second terminal with `cd frontend` followed by `npm run dev`. Upload the sample policy PDF and ask a question that produces two facts. Expand each claim's evidence list: each list should come from that claim's own query embedding and FAISS search, while the “Retrieved passages” panel continues to show the separate question-level search.

Run `python -m pytest backend/tests -q` from the root to exercise the retrieval and verification behavior with fake models. Run `npm run build` from `frontend` to check the interface. These checks do not establish that the methods are accurate; that requires annotated evidence and a controlled evaluation.

