# Evaluating Board Intelligence

The [fictional corpus](../../examples/board_packs.json) contains three packs with three pages each. The [golden set](golden_set.json) annotates nine answerable questions, three unanswerable questions, and 12 governance claims. Eight of the replay claims reflect legitimate decisions or advisory actions. Four transfer authority from the board to a committee.

```powershell
python -m tests.rag_eval.eval_runner --guard-only
python -m tests.rag_eval.eval_runner
```

The first command replays fixed, labeled claims directly through the rules using matching source passages, with the same claims serving as the no-rules baseline. It reports true and false positives and negatives, precision, and recall. It runs without GPU weights or an API.

The second command also generates and uploads the fictional PDFs to the local API. It starts a fresh session for each question, records the retrieved passages, answer, verification status, and timings, and deletes the session afterward. The first-chunk timing includes retrieval and measures the first SSE payload. The response timing includes verification.

The grader checks each generated claim for a citation and looks for a matching annotated fact on every named page. Results include the claim text and pages for manual review. Matching uses regular expressions, so uncommon valid paraphrases and additional correct facts can receive no credit. These are synthetic regression checks on short, text-native PDFs. They provide a starting point for assessing the system on larger packs and scanned material.

The default result path is `tests/rag_eval/results/latest.json`. Use `--output` to save separate runs.

## Reference run

The [per-question output](reference-rtx5090.json) comes from one local run on an RTX 5090 Laptop GPU with 24 GB VRAM, Windows and WSL2, Qwen3-14B-AWQ, and vLLM 0.23.0. Median time to the first streamed chunk was 2.21 seconds. Median time through verification was 4.50 seconds. The grader gave credit for all nine answerable cases and the model abstained on all three unanswerable cases. Verification flagged ME-03 for low entailment despite its matching the annotated fact and page.
