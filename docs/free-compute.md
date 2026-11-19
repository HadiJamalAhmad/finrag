# Free compute for finrag

Every heavy step of finrag (an LLM over many questions, indexing, a bake-off) can run on free
online compute, so nothing depends on a local machine. One rule for all of them: a run reads
the frozen corpus in `eval/fixtures/fr-interieur/docs`, checks the model against its pinned
digest, and writes a report to `eval/reports/` that is then committed or attached.

| Option | What it gives | Used here for | Status |
|---|---|---|---|
| GitHub Actions | Free minutes for public repositories; Linux runner, 4 CPUs, 16 GB, 6 h per job, no GPU | `extraction-eval`, `heldout-eval` (on demand), CI on every push | **Ready** |
| Kaggle Notebooks | Free GPU (T4 x2 or P100), weekly quota, 12 h sessions | `deploy/notebooks/gpu_eval.ipynb`: same evaluations in minutes, larger models | **Ready** (needs a Kaggle account) |
| Google Colab | A T4 when available, sessions end when idle | Same notebook, quick checks | **Ready** (needs a Google account) |
| Modal | Monthly free credits | Live API behind finrag-ten.vercel.app, GPU bake-off | Waiting for credits (repository variable `MODAL_DEPLOY=on`) |
| Oracle Cloud Always Free | ARM VM, 4 OCPUs, 24 GB RAM, no time limit | A permanent demo: API, Qdrant, llama3.2:3b on CPU | To set up (needs an account) |
| Hugging Face Spaces | Static Spaces only (Docker and Gradio Spaces on CPU now need PRO) | A public demo: search and refusals run in the browser | **Ready**: `static-space` workflow |

Quotas change: check the provider's page before relying on a number above.

## GitHub Actions (ready)

Both workflows run only when started by hand, so no minute is spent by accident.

```bash
gh workflow run extraction-eval.yml          # fields + proof sentences, 14 sheets
gh workflow run heldout-eval.yml             # never-seen questions, settings frozen
gh run watch                                 # then download the artifact
gh run download --name extraction-report     # or heldout-reports
```

On a CPU runner the 3B model answers in tens of seconds, so a full run takes from a few minutes
to about an hour. The job log and the step summary are public and dated.

## Kaggle and Colab (ready, need an account)

Open the notebook straight from GitHub:

- Colab: `https://colab.research.google.com/github/HadiJamalAhmad/finrag/blob/main/deploy/notebooks/gpu_eval.ipynb`
- Kaggle: Create > New Notebook > File > Import Notebook > GitHub, then this repository and
  `deploy/notebooks/gpu_eval.ipynb`. Turn on the GPU and Internet in the settings panel.

Set `MODEL` in the first cell (`llama-3b` or `qwen3-4b` on a 16 GB GPU), run all cells, download
the zip of `eval/reports/` and commit the reports you want to keep.

## Modal (when credits are available)

Unchanged: set the repository variable `MODAL_DEPLOY=on` and push to `main`; the CI deploys the
backend and the Vercel page talks to it again. Set it back to off when credits run low.

## Oracle Cloud Always Free (to set up)

1. Create an account (a card is asked for identity; Always Free resources are not charged).
2. Create a VM: shape `VM.Standard.A1.Flex`, 4 OCPUs, 24 GB, Ubuntu 24.04, open port 443.
3. Install Docker, then run the stack of `deploy/k8s` with k3s, or Docker Compose with the
   same images (API, Qdrant, Ollama pinned to `ollama/ollama:0.12.6`).
4. `finrag pull-model` (digest checked) and `finrag ingest` against the frozen corpus.

## Hugging Face Spaces (static, free)

Docker and Gradio Spaces on the free CPU now need a PRO subscription; static Spaces stay free.
So the demo runs in the visitor's browser: the `static-space` workflow computes the 598
passages' vectors with finrag's own `embed()`, then checks that the browser model
(transformers.js, `Xenova/paraphrase-multilingual-MiniLM-L12-v2` at a pinned revision) gives the
same answer/refuse decision as finrag on all 69 evaluation questions, at q8, fp16 and fp32, and
keeps the lightest precision that agrees AND whose largest score error stays below the
closest question's distance to the threshold (q8 agreed on all 69 but erred by 0.030 against a
margin of 0.004, so fp16 is kept). No such precision, nothing to publish.

```bash
gh workflow run static-space.yml && gh run watch
gh run download --name static-space -D /tmp/space
hf upload HadiJamalAhmad/finrag /tmp/space/site . --repo-type space
```
