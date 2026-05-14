---
title: "Falsification-Driven Reinforcement Learning for Maritime Motion Planning"
date: 2025-04-10
math: true
content_public: false
venue: "Ocean Engineering"

pdf: ""
arxiv: "https://arxiv.org/abs/2510.06970"
code: "https://github.com/marlonmueller/fdrl-maritime"

authors:
  - name: "Marlon Müller*"
    url: "https://marlonmueller.github.io/"
    affiliation: "1,2,3"
  - name: "Florian Finkeldei*"
    url: ""
    affiliation: "1"
  - name: "Hanna Krasowski*"
    url: "https://hanna.krasowski.io/"
    affiliation: "2"
  - name: "Murat Arcak"
    url: "https://people.eecs.berkeley.edu/~arcak/"
    affiliation: "2"
  - name: "Matthias Althoff"
    url: "https://www.ce.cit.tum.de/cps/members/prof-dr-ing-matthias-althoff/"
    affiliation: "1,3"

affiliations:
  - num: "1"
    name: "Technical University of Munich"
  - num: "2"
    name: "University of California, Berkeley"
  - num: "3"
    name: "Munich Center for Machine Learning (MCML)"

abstract: >
  Compliance with maritime traffic rules is essential for the safe operation of autonomous vessels, yet training
  reinforcement learning (RL) agents to adhere to them is challenging. The behavior of RL agents is shaped by the
  training scenarios they encounter, but creating scenarios that capture the complexity of maritime navigation
  is non-trivial, and real-world data alone is insufficient. To address this, we propose a falsification-driven RL
  approach that generates adversarial training scenarios in which the vessel under test violates maritime traffic
  rules, which are expressed as signal temporal logic specifications. Our experiments on open-sea navigation with
  two vessels demonstrate that the proposed approach provides more relevant training scenarios and achieves
  more consistent rule compliance.

bibtex: |
  @article{muller2025falsification,
  title={Falsification-Driven Reinforcement Learning for Maritime Motion Planning},
  author={M{\"u}ller, Marlon and Finkeldei, Florian and Krasowski, Hanna and Arcak, Murat and Althoff, Matthias},
  journal={arXiv preprint arXiv:2510.06970},
  year={2025}
  }

references:
  - text: 'H. Krasowski and M. Althoff. "Temporal Logic Formalization of Marine Traffic Rules." *IEEE Intelligent Vehicles Symposium*, pp. 186–192, 2021.'
    url: "https://doi.org/10.1109/IV48863.2021.9575685"
  - text: 'T. Ferrère, D. Nickovic, A. Donzé, H. Ito, and J. Kapinski. "Interface-aware signal temporal logic." *ACM HSCC*, pp. 57–66, 2019.'
    url: "https://doi.org/10.1145/3302504.3311800"
  - text: 'J. Schulman et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017.'
    url: "https://arxiv.org/abs/1707.06347"
  - text: 'N. Hansen. "The CMA Evolution Strategy: A Tutorial." *arXiv:1604.00772*, 2016.'
    url: "https://arxiv.org/abs/1604.00772"
  - text: 'J. Kennedy and R. Eberhart. "Particle Swarm Optimization." *ICNN*, 1995.'
    url: "https://doi.org/10.1109/ICNN.1995.488968"
  - text: 'S. Huang et al. "CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms." *JMLR* 23(274), 2022.'
    url: "https://jmlr.org/papers/v23/21-1342.html"
---
