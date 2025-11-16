## Executed Lab 1 with the following modifications to the LLM data pipeline:

- Added an explicit train–validation split on WikiText-2 so the tokenization and sequence-building pipeline produces both training and held-out evaluation data.
- Changed the sequence-grouping step from simple non-overlapping blocks to a sliding-window chunking strategy with overlap, which is a common technique in LLM pretraining to expose tokens to multiple contexts.
- Implemented a custom collate_fn that pads sequences, builds an attention_mask, and masks padding tokens in the labels with -100, matching the expected input format of Hugging Face causal language models.
- Added basic token-length statistics on the tokenized dataset (mean, median, percentiles) to justify the chosen context window size and to inspect the distribution of sequence lengths in the corpus.
