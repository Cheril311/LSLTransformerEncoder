# Language Specific Layer Transformer

This repository contains the implementation of a multilingual transformer with language-specific encoder layers as described in https://aclanthology.org/2023.acl-long.825/.

## Setup

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.

## Structure

- `models/`: Contains the model definitions.
- `data/`: Place your datasets by making this directory.
- `scripts/`: You can place your training scripts by making this directory.
- `checkpoints/`: Place your saved model weights by making this directory.

## Citation

If you use this code in your research, please cite the following paper:
<pre>
@inproceedings{pires-etal-2023-learning,
title = "Learning Language-Specific Layers for Multilingual Machine Translation",
author = "Pires, Telmo and
Schmidt, Robin and
Liao, Yi-Hsiu and
Peitz, Stephan",
editor = "Rogers, Anna and
Boyd-Graber, Jordan and
Okazaki, Naoaki",
booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
month = jul,
year = "2023",
address = "Toronto, Canada",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2023.acl-long.825",
doi = "10.18653/v1/2023.acl-long.825",
pages = "14767--14783",
abstract = "Multilingual Machine Translation promises to improve translation quality between non-English languages. This is advantageous for several reasons, namely lower latency (no need to translate twice), and reduced error cascades (e.g., avoiding losing gender and formality information when translating through English). On the downside, adding more languages reduces model capacity per language, which is usually countered by increasing the overall model size, making training harder and inference slower. In this work, we introduce Language-Specific Transformer Layers (LSLs), which allow us to increase model capacity, while keeping the amount of computation and the number of parameters used in the forward pass constant. The key idea is to have some layers of the encoder be source or target language-specific, while keeping the remaining layers shared. We study the best way to place these layers using a neural architecture search inspired approach, and achieve an improvement of 1.3 chrF (1.5 spBLEU) points over not using LSLs on a separate decoder architecture, and 1.9 chrF (2.2 spBLEU) on a shared decoder one.",
}
</pre>
