# In Search of Hallucinations: GPT-2 Attention Analysis

Companion notebook to the blog series [In Search of Hallucinations](https://rfischer.com) at rfischer.com. The series documents a hands-on investigation into whether internal model signals -- attention entropy, Gini coefficient, head disagreement, and hidden state variance -- can detect hallucinations at inference time without ground truth labels.

The short answer is that no reliable signal was found. The notebook is published anyway because the methodology, the things that went wrong, and the reasons they went wrong are worth documenting.

## What This Is

The investigation targets a specific and hard problem: detecting subtle, plausible falsehoods of the kind that appear in real hallucinations ("Bell invented the telegraph"), not obvious nonsense. Obvious errors produce measurable signals; the question is whether any signal survives on content that a model treats as coherent and plausible.

Experiments were run on GPT-2 Medium (345M parameters) as the primary model, with replication runs on GPT-2 Large (774M) and Qwen2.5-0.5B. The notebook covers:

- Plausible false pairs: true and false statements on the same topic, tested for structural differences in attention and hidden state metrics
- Probability and confidence analysis: whether output token probabilities distinguish true from false completions
- Organic failures: math problems GPT-2 cannot solve, providing ground-truth hallucination scenarios
- Baseline and difficulty gradient: entropy across easy questions, simple arithmetic, and hard multiplication
- Long-form generation: real-time per-token entropy tracking during multi-sentence generation
- Continuation cascade: whether a false premise early in a sequence degrades subsequent signals
- Paraphrasing robustness: whether signals are consistent across different phrasings of the same statement
- Cross-architecture replication: GPT-2 Large and Qwen2.5 runs

Each section includes the hypothesis, what was measured, and what the result actually showed.

## Key Findings

None of the metrics produced a reliable hallucination detector:

- Entropy differences in plausible pairs tracked statement position, not factual accuracy (position bias)
- Gini and hidden state variance signals appeared on semantically incoherent content, suggesting they reflect something other than factual reasoning
- Entropy was higher on easy questions than on hard math, opposite to what a difficulty-tracking hypothesis would predict
- GPT-2 Large replicated the reversed entropy pattern rather than correcting it
- Continuation cascade showed false premises producing more semantically consistent continuations than true ones
- Paraphrasing robustness was at chance level across the five pairs tested
- Qwen2.5 showed consistently stable entropy regardless of content accuracy

One partial result: long-form entropy thresholding flagged tokens at a rate that might be informative, but the threshold was not discriminating enough to separate hallucinated from accurate content reliably.

## Running the Notebook

Kaggle is the recommended environment. The notebook is structured to run top-to-bottom on a standard Kaggle GPU session. Most dependencies are pre-installed; the install cell at the top handles anything that is not.

To try it locally, you will need Python 3.9-3.12, PyTorch, and the Hugging Face `transformers` library. GPT-2 weights download automatically. Qwen2.5 is larger and may require more memory than a typical laptop has available.

The notebook uses `raise SystemExit()` at the end of sections that are meant to be run independently. Comment those out if you want to run everything in sequence.

## Caveats Worth Knowing Before You Vary the Experiments

GPT-2 is not a modern model. Signals that do not appear here might appear at scale, or on models with different training objectives. Detection thresholds calibrated on GPT-2 would need recalibration on any other model.

Qwen2.5 entropy is normalized to 0-1. GPT-2 entropy is in raw nats. Numerical comparisons across models are not valid.

GPT-2 enters repetition loops at higher token counts, which artificially inflates entropy measurements in the long-form sections. This is noted in the relevant cells.

The plausible pairs dataset is small (around 30 pairs across three categories). Any result on that dataset should be treated with caution before generalizing.

## Blog Series

The full narrative, including why each experiment was designed the way it was and what was learned from the failures, is at rfischer.com. The blog and the notebook are intended to be read together; the blog explains the reasoning and the notebook provides the code to reproduce or vary the experiments.
