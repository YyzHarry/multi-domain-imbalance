# On Multi-Domain Long-Tailed Recognition, Generalization and Beyond

This repository contains the implementation code for paper: [On Multi-Domain Long-Tailed Recognition, Generalization and Beyond](https://arxiv.org/abs/xxx).

<p align="center">
    <img src="mdlt.png" width="600"> <br>
<b>Multi-Domain Long-Tailed Recognition (MDLT)</b> aims to learn from imbalanced data from distinct domains, <br> tackle potential missing data for certain regions, and generalize to the entire target range.
</p>


## Beyond Single-domain Imbalance: Brief Introduction for MDLT
Existing techniques for learning from imbalanced data focus on targets with __categorical__ indices, i.e., the targets are different classes. However, many real-world tasks involve __continuous__ and even infinite target values. We systematically investigate _Deep Imbalanced Regression (DIR)_, which aims to learn continuous targets from natural imbalanced data, deal with potential missing data for certain target values, and generalize to the entire target range.

We curate and benchmark large-scale DIR datasets for common real-world tasks in _computer vision_, _natural language processing_, and _healthcare_ domains, ranging from single-value prediction such as age, text similarity score, health condition score, to dense-value prediction such as depth.


## Domain Generalization under Data Imbalance
In additional to MDLT, we show that 


## Updates
- __[03/17/2022]__ [ArXiv version](https://arxiv.org/abs/xxx) posted. The code is currently under cleaning. Please stay tuned for updates.


## Citation
```bib
@article{yang2022multi,
  title={On Multi-Domain Long-Tailed Recognition, Generalization and Beyond},
  author={Yang, Yuzhe and Wang, Hao and Katabi, Dina},
  journal={arXiv preprint arXiv:xxx},
  year={2022}
}
```
