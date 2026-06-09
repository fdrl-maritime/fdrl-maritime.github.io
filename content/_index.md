---
title: "Falsification-Driven Reinforcement Learning for Maritime Motion Planning"
date: 2025-04-10
math: true
content_public: true
venue: "Ocean Engineering"

pdf: "https://www.sciencedirect.com/science/article/pii/S0029801826014137"
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
  @article{Mueller2026.falsificationDrivenRLMaritime,
  author = {Marlon M{\"u}ller and Florian Finkeldei and Hanna Krasowski and Murat Arcak and Matthias Althoff},
  title = {Falsification-driven reinforcement learning for maritime motion planning},
  journal = {Ocean Engineering},
  volume = {361},
  pages = {125579},
  year = {2026},
  issn = {0029-8018},
  doi = {10.1016/j.oceaneng.2026.125579},
  url = {https://www.sciencedirect.com/science/article/pii/S0029801826014137},
  }

references:
  - text: 'International Maritime Organization, 1972. COLREGs: Convention on the international regulations for preventing collisions at sea.'
    url: "https://www.imo.org/en/about/conventions/pages/colreg.aspx"
  - text: 'Krasowski, H., Althoff, M., 2024. Provable traffic rule compliance in safe reinforcement learning on the open sea. IEEE Trans. Intell. Veh. 9, 7617–7634.'
    url: "https://ieeexplore.ieee.org/document/10530091"
  - text: 'Ferrère, T., Nickovic, D., Donzé, A., Ito, H., Kapinski, J., 2019. Interface-aware signal temporal logic. In: ACM International Conference on Hybrid Systems: Computation and Control. pp. 57–66.'
    url: "https://dl.acm.org/doi/10.1145/3302504.3311800"
  - text: 'Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., van Zee, M., 2024. Flax: A neural network library and ecosystem for JAX.'
    url: "http://github.com/google/flax"
  - text: 'Huang, S., Dossa, R.F.J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., Araújo, J.G., 2022. CleanRL: High-quality single-file implementations of deep reinforcement learning algorithms. J. Mach. Learn. Res. 23 (274), 1–18.'
    url: "https://github.com/vwxyzjn/cleanrl"
  - text: 'Lange, R.T., 2023. evosax: JAX-based evolution strategies. In: Genetic and Evolutionary Computation Conference Companion. pp. 659–662.'
    url: "https://github.com/RobertTLange/evosax"
---

## Overview

In this article, we propose a falsification-driven training approach based on signal temporal logic (STL) specifications of maritime traffic rules. The approach generates adversarial scenarios in which rule violations occur and incorporates them into the training process. This systematically exposes the controller to safety-critical encounters (see Fig. 1).

{{< figure
  src="/images/falsification_driven_rl/framework_overview.svg"
  alt="Overview of the falsification-driven RL framework"
  width="65%"
  iw="1991"
  ih="1546"
  caption="Every training iteration, the falsification module finds scenarios where the ego policy violates the traffic rule specification $\varphi$. These adversarial scenarios are pooled and replayed during PPO training."
>}}

## Convention on the International Regulations for Preventing Collisions at Sea

We adopt the formalization of the COLREGs {{< cite 1 >}} from {{< cite 2 >}} and extend it with robustness measures to enable falsification. All encounter types share a common structure: a maneuver obligation is triggered when a persistent encounter is active, and compliance requires both timely maneuvering and resolution of the collision risk within specified time bounds:

$$
\begin{equation}
\begin{aligned}
\varphi_{\texttt{colregs}} \;\mathrel{:=}\; & \mathbf{G}_{[0,\,\Delta t N]}\Bigl(\texttt{persistent\_encounter}(\mathbf{x}^\mathrm{E}, \mathbf{x}^\mathrm{A}, k) \;\Longrightarrow \\
&  \mathbf{F}_{[t_\mathrm{p},\, t_\mathrm{p}+t_\mathrm{m}]}\!\left(\texttt{maneuver\_encounter}(\mathbf{x}^\mathrm{E}, k)\right) \;\wedge\\
& \mathbf{F}_{[t_\mathrm{p},\, t_\mathrm{p}+2t_\mathrm{m}]}\!\left(\neg\,\texttt{velocity\_obstacle}(\mathbf{x}^\mathrm{E}, \mathbf{x}^\mathrm{A}, k)\right)\Bigr)\;.
\end{aligned}
\end{equation}
$$

The high-level predicates in (1) are built from six atomic predicates. For example, $\texttt{position\_halfplane}$ (see Fig. 2a) and $\texttt{orientation\_halfplane}$ (see Fig. 2b) define half-planes relative to the ego vessel. Conjunctions of these half-planes are used to detect whether another vessel lies in a specific angular sector or heading range relative to the ego.

STL admits quantitative semantics that assigns a robustness value to each predicate, which makes it possible to frame rule compliance testing as an optimization problem that minimizes the robustness of the specification. The robustness measures $h_\texttt{position\_halfplane}$ and $h_\texttt{orientation\_halfplane}$ are the signed distances to the half-planes, rescaled to the time domain.

{{< figure
  src="/images/falsification_driven_rl/atomic_predicates.svg"
  alt="Illustration of the position_halfplane and orientation_halfplane robustness measures"
  width="48%"
  iw="1602"
  ih="972"
  caption="Illustration of atomic predicates and robustness measures (a) $h_\texttt{p} \mathrel{:=} h_\texttt{position\_halfplane}$ and (b) $h_\texttt{o} \mathrel{:=} h_\texttt{orientation\_halfplane}$ with $\beta, \gamma \mathrel{:=} \pi/4$."
>}}

## Interface-aware Signal Temporal Logic

The specification (1) has an implication structure: the maneuver obligation only applies when an encounter is active. The common robustness semantics for STL collapse the antecedent and consequent into a single scalar value. A high robustness therefore does not necessarily indicate good maneuvering behavior. For example, a scenario where the vessels never come close to each other would yield a high robustness score due to vacuous satisfaction of the implication, even though the policy has not learned to maneuver at all.

Interface-aware STL (IA-STL) {{< cite 3 >}} addresses this by tracking the addresses this issue by evaluating the robustness of the
antecedent and the consequent separately. The input robustness $\rho_\mathrm{in}$ quantifies how far the scenario is from activating the encounter condition, while the output robustness $\rho_\mathrm{out}$ quantifies how well the maneuver obligation is satisfied once active. We use IA-STL to define a falsification objective that first drives the scenario towards an encounter and then minimizes the output robustness to expose a violation.

## Experimental Setup

The approach is implemented in Flax {{< cite 4 >}}, and based on CleanRL {{< cite 5 >}}. The falsification optimizers are provided by evosax {{< cite 6 >}}. Since critical encounter scenarios are rare in maritime data, we generate random scenarios to train a baseline policy. The adversary control inputs are sampled from a mean-reverting stochastic process. For the baseline, new scenarios are randomly sampled at each episode. We evaluate on two scenario distributions: a baseline evaluation distribution and a falsification evaluation distribution. The baseline evaluation distribution equals the training distribution of the baseline policy. The falsification evaluation distribution consists of scenarios generated by the falsification module throughout training.

## Results

On the baseline evaluation distribution, the baseline policy shows the highest goal-reaching rate (see Table 1). Compliance, including vacuous encounters, is above 95% for all methods. However, compliance on nonvacuous encounters reduces to 78.3% for the baseline, and remains more stable for the falsification-driven policies with at least 89.2%. The proportion of nonvacuous scenarios, the so-called engagement, is lower for the baseline compared to the falsification-driven approaches. Note that the baseline evaluation distribution used to
compute the metrics in Table 1 differs from the scenario distributions used to train
the falsification-driven agents.

<div class="table-scroll">
<table>
<caption><strong>Table 1.</strong> Results on the baseline evaluation distribution. †All scenarios; ‡non-vacuous (engaged) scenarios only. Mean ± std over 10 seeds.</caption>
<thead>
<tr>
<th>Metric</th>
<th style="text-align:right">Baseline</th>
<th style="text-align:right">CMA-ES</th>
<th style="text-align:right">PSO</th>
<th style="text-align:right">SA</th>
</tr>
</thead>
<tbody>
<tr><td>Goal (%)</td><td style="text-align:right">75.1 ± 2.0</td><td style="text-align:right">45.6 ± 6.9</td><td style="text-align:right">46.7 ± 8.1</td><td style="text-align:right">44.3 ± 6.1</td></tr>
<tr><td>Compliance† (%)</td><td style="text-align:right">95.7 ± 1.0</td><td style="text-align:right">96.3 ± 1.1</td><td style="text-align:right">96.6 ± 1.0</td><td style="text-align:right">96.4 ± 0.9</td></tr>
<tr><td>Robustness $\rho_\mathrm{in}$</td><td style="text-align:right">11.2 ± 2.7</td><td style="text-align:right">6.0 ± 1.2</td><td style="text-align:right">6.4 ± 0.8</td><td style="text-align:right">7.4 ± 2.4</td></tr>
</tbody>
<tbody class="group-b">
<tr><td>Engagement (%)</td><td style="text-align:right">21.0 ± 5.6</td><td style="text-align:right">37.6 ± 5.2</td><td style="text-align:right">33.9 ± 3.9</td><td style="text-align:right">33.1 ± 6.4</td></tr>
<tr><td>Compliance‡ (%)</td><td style="text-align:right">78.3 ± 6.3</td><td style="text-align:right">90.0 ± 2.7</td><td style="text-align:right">89.8 ± 3.9</td><td style="text-align:right">89.2 ± 1.8</td></tr>
<tr><td>Robustness $\rho_\mathrm{out}$</td><td style="text-align:right">10.9 ± 2.2</td><td style="text-align:right">15.4 ± 3.0</td><td style="text-align:right">15.2 ± 2.6</td><td style="text-align:right">16.8 ± 3.8</td></tr>
</tbody>
</table>
</div>

On the falsification evaluation distribution (see Table 2), the goal-reaching rate decreases for all methods. For the baseline policy, excluding vacuous encounters, the compliance of the baseline agent reduces to 35.9%. The falsification-driven policies show higher engagement rates and maintain significantly higher compliance of at least 86% when including vacuous encounters and 72% when excluding them.

<div class="table-scroll">
<table>
<caption><strong>Table 2.</strong> Results on the falsification evaluation distribution. †All scenarios; ‡non-vacuous (engaged) scenarios only. Mean ± std over 10 seeds.</caption>
<thead>
<tr>
<th>Metric</th>
<th style="text-align:right">Baseline</th>
<th style="text-align:right">CMA-ES</th>
<th style="text-align:right">PSO</th>
<th style="text-align:right">SA</th>
</tr>
</thead>
<tbody>
<tr><td>Goal (%)</td><td style="text-align:right">30.6 ± 2.0</td><td style="text-align:right">27.4 ± 5.3</td><td style="text-align:right">26.3 ± 5.8</td><td style="text-align:right">27.0 ± 4.9</td></tr>
<tr><td>Compliance† (%)</td><td style="text-align:right">57.6 ± 2.1</td><td style="text-align:right">86.8 ± 1.3</td><td style="text-align:right">86.7 ± 1.8</td><td style="text-align:right">86.7 ± 1.4</td></tr>
<tr><td>Robustness $\rho_\mathrm{in}$</td><td style="text-align:right">−7.2 ± 0.8</td><td style="text-align:right">−0.2 ± 1.4</td><td style="text-align:right">0.3 ± 1.2</td><td style="text-align:right">0.5 ± 1.8</td></tr>
</tbody>
<tbody class="group-b">
<tr><td>Engagement (%)</td><td style="text-align:right">66.1 ± 2.5</td><td style="text-align:right">49.8 ± 4.4</td><td style="text-align:right">47.5 ± 2.7</td><td style="text-align:right">47.1 ± 4.2</td></tr>
<tr><td>Compliance‡ (%)</td><td style="text-align:right">35.9 ± 2.8</td><td style="text-align:right">73.3 ± 3.3</td><td style="text-align:right">72.0 ± 3.8</td><td style="text-align:right">71.7 ± 1.4</td></tr>
<tr><td>Robustness $\rho_\mathrm{out}$</td><td style="text-align:right">−2.1 ± 1.9</td><td style="text-align:right">19.2 ± 3.0</td><td style="text-align:right">19.8 ± 2.5</td><td style="text-align:right">20.7 ± 3.0</td></tr>
</tbody>
</table>
</div>

The results illustrate
the underlying trade-off between rule compliance and goal-reaching. The agent should neither avoid encounters persistently nor
deliberately induce them, as both strategies can compromise its ability
to reach the goal. At the same time, prioritizing correct maneuvering
over purely goal-directed behavior is necessary to ensure compliance
with maritime traffic rules. Our experiments indicate that incorporating falsification improves the training process by generating more relevant scenarios, which in turn leads to more consistent rule compliance. These findings highlight the potential of combining formal methods with learning-based control to enhance the safety of autonomous systems.

For a comprehensive presentation of the approach please refer to the full article **[here](https://www.sciencedirect.com/science/article/pii/S0029801826014137)**.
