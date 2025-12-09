# Information Integration Window in LLMs
Code and data for the paper "Information Integration in Large Language Models is Gated by Linguistic Structural Markers". See [the paper](https://aclanthology.org/2025.emnlp-main.351.pdf).

## Prerequisites
Python 3.9.

## Install & Getting Started

1. Clone the repository.

2. Install the packages.

```bash
pip install -r requirements.txt
```

3. Run `bash run.sh` to replicate the experiment in the paper.

## Visualization

See the code in `./plot/analysis.ipynb`.

## Citation

If you make use of the code in this repository, please cite the following papers:

```
@inproceedings{liu-ding-2025-information,
    title = "Information Integration in Large Language Models is Gated by Linguistic Structural Markers",
    author = "Liu, Wei  and
      Ding, Nai",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.351/",
    doi = "10.18653/v1/2025.emnlp-main.351",
    pages = "6903--6915",
    ISBN = "979-8-89176-332-6",
    abstract = "Language comprehension relies on integrating information across both local words and broader context. We propose a method to quantify the information integration window of large language models (LLMs) and examine how sentence and clause boundaries constrain this window. Specifically, LLMs are required to predict a target word based on either a local window (local prediction) or the full context (global prediction), and we use Jensen-Shannon (JS) divergence to measure the information loss from relying solely on the local window, termed the local-prediction deficit. Results show that integration windows of both humans and LLMs are strongly modulated by sentence boundaries, and predictions primarily rely on words within the same sentence or clause: The local-prediction deficit follows a power-law decay as the window length increases and drops sharply at the sentence boundary. This boundary effect is primarily attributed to linguistic structural markers, e.g., punctuation, rather than implicit syntactic or semantic cues. Together, these results indicate that LLMs rely on explicit structural cues to guide their information integration strategy."
}
```
