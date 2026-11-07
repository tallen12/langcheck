---
name: model-trace
description: Trace the local tagger and proposer sentence by sentence on a judged clean set and build an HTML page of every stage (gold edits with judge verdicts, flags and scores, the proposer's beam and rerank, filter fates, output) plus where each missed fix was lost. Use when asked why the model misses or invents corrections, to inspect a model on judged data, or to make a trace page or artifact for a checkpoint.
---

# Model trace

Shows, for every sentence of a judged clean set, what the tagger flagged, what the
proposer wrote for each flag, how reranking ordered it, what the filters and gate did
with the top proposals, what was offered, and how that scored. Each reference edit is
labelled with the stage where the model lost it, so a run's misses split into
*not flagged*, *flagged but never proposed*, *in the beam below the top 8* and
*proposed, then filtered out*.

The trace uses the settings of `langcheck suggest measure-file`, so its totals must match
that command's report on the same file. Check this every time (step 4).

## 1. Judge the set and write a clean set

Skip what already exists under `data/judgments/<set>/`.

```bash
uv run langcheck judge run cweb-g-dev --limit 300
uv run langcheck judge clean cweb-g-dev --limit 300
```

`judge run` re-judges only what is missing, so rerun it once if it reports failures.
About 200 new sentences take an hour with the default panel. `judge clean` writes
`clean-local.jsonl` (sentences that need a linked edit are left out) and prints how many
sentences it kept and why others are out. Pass the same `--limit` to both.

## 2. Free the GPU

The LLM server and the tagger share the GPU when the server runs on this machine. Loaded
judge models leave no room for the tagger, which then fails with a CUDA out-of-memory
error. With Ollama, list what is loaded and unload anything idle:

```bash
curl -s <server>/api/ps
curl -s <server>/api/generate -d '{"model": "<name>", "keep_alive": 0}'
```

`<server>` is `llm.base_url` from `uv run langcheck config show` without the `/v1`.
Unload only models nothing else is using.

## 3. Score the model

```bash
uv run langcheck suggest measure-file <tagger> data/judgments/cweb-g-dev/clean-local.jsonl \
    --limit 1000 --proposer <proposer> --output runs/judged/clean-local.json
```

Precision, recall and F0.5 count edits, not sentences. Sentences that need no change
and get none count nowhere. When a sentence has several valid references, ERRANT scores
it against the one that suits the model best, so the score's reference edits can be
fewer than the main references hold.

## 4. Trace and check the totals

```bash
uv run python .agents/skills/model-trace/scripts/trace.py \
    --tagger <tagger> --proposer <proposer> --dataset cweb-g-dev --limit 300 \
    --out runs/judged/trace.json
```

The last lines print the matched, false-alarm and missed counts (`tp fp fn`), the main
and scored reference edits, and the stage counts. Matched, false alarms and missed must
equal the `filtered-first` row of step 3's report. If they differ, the trace no longer
follows the pipeline: fix the script before you build anything from it. `--clean`,
`--data` and `--store` default to the usual paths under `data/`. The
run takes about five minutes for 274 sentences on a GPU.

## 5. Build the page

```bash
uv run python .agents/skills/model-trace/scripts/build_page.py runs/judged/trace.json \
    --out runs/judged/trace.html
```

The page is one self-contained HTML file (about 650 KB for 274 sentences). It holds:

- the score and a bar for each stage where fixes were lost;
- why the main and scored reference edit counts differ;
- filters for false alarms and for each kind of miss, plus a text search;
- one entry per sentence, as described below.

Each sentence entry shows:

- the sentence, with each flag shaded by its score;
- each annotator's edits, with the judges' verdict and votes, and linked edits marked;
- the clean references, with the one scored against marked and each edit's stage;
- each flag's beam in reranked order, with log probability, tagger drop, rerank score,
  how much the correction lowers the sentence's score, and its filter result. The top 8
  are marked; they are the proposals that reach the filters.
- the model's output, marked up against the sentence.

## 6. Share it privately

The page contains the evaluation set's sentences. CWEB is CC BY-NC-SA, so never commit
the trace or the page (`runs/` and `data/` are ignored), and share the page only
privately. To publish it as a hosted page, use the HTML file as is; it loads only Google
Fonts.

## Notes

- Filter results are tied to beam rows by their rank after reranking. Do not match them
  by text: for an insertion, the filters see the placed text with its spacing (`" your"`),
  not the beam's `"your"`.
- `trace.py` records reranking through the `observer` of `ProposerCorrector`
  (`RerankObserver` in `src/langcheck/proposal/protocols.py`), so it runs the real
  pipeline rather than a copy. It runs one sentence at a time so each rerank belongs to one
  flag.
- To trace another checkpoint or set, change `--tagger`, `--proposer`, `--dataset` and
  `--limit`. The set must have judgments from the default panel.
