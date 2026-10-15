# Evaluation Protocol

1. Upload a fixed set of publicly distributable PDFs.
2. Run the app once to index documents, or start the API to rebuild indexes.
3. Inspect `data/corpus.json` and manually label *all* relevant chunk IDs for each question.
4. Save real questions in `data/benchmark_questions.json` using the structure in `data/benchmark_questions.example.json`.
5. Run `python -m researchpilot.evaluation.benchmark`.
6. Report Recall@5, MRR@5, mean latency, corpus size, embedding model, chunk parameters, and hardware.

Do not report example data as measured results. Compare retrieval separately from final answer accuracy. Note that manually incomplete relevance judgments can bias Recall@K.
