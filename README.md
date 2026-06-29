# LieFlow: Discovering Symmetry Groups with Flow Matching

This repo is the official implementation of the ICML 2026 paper [Discovering Symmetry Groups with Flow Matching](https://arxiv.org/abs/2512.20043).

Project website: <https://jypark0.github.io/lieflow/>

## 1. Paper summary

LieFlow reframes symmetry discovery as **distribution learning on a Lie
group**. Given a hypothesis Lie group `G` and unlabeled data `x ~ p(x)`, we
learn a conditional distribution `q_θ(g | x)` over group elements `g ∈ G` whose
support concentrates on the underlying symmetry subgroup `H ⊆ G`. Training
combines a flow matching objective on the Lie algebra with a power time-sampling
schedule that focuses gradients near `t ≈ 1`, mitigating the "last-minute mode
convergence" problem on discrete subgroups.

## 2. Installation

Requires Python 3.11+ (tested on 3.11).

```bash
pip install -e .
```

This installs the `lieflow` package and all dependencies (see `pyproject.toml`).
Saving the 2D progression animation (§4.1) additionally requires
[`ffmpeg`](https://ffmpeg.org/) installed and on your `PATH`.
The code uses [Hydra](https://hydra.cc/) for configuration management and
[Weights & Biases](https://wandb.ai/) for experiment tracking; W&B can be
disabled with `WANDB_MODE=offline`.

## 3. Running experiments

All training scripts share the same Hydra structure:

```bash
python experiments/${EXP_FILE} \
    dataset=${DATASET} \
    model=${MODEL} \
    seed=${SEED}
```

where:

- `${EXP_FILE}` is one of the scripts under `experiments/`,
- `${DATASET}` is a YAML in `conf/dataset/`,
- `${MODEL}` is a YAML in `conf/model/<family>/`.

Outputs (logs, checkpoints, figures) land in `outputs/${date}/${time}/`
(the default Hydra layout).

## 4. Experiments

### 4.1 2D arrow

To run the synthetic 2D datasets, use `flow_matching_2d.py` as the `EXP_FILE`
with a different `MODEL` for each hypothesis group. For sampling group elements
from `GL(2, R)+`, the paper (Appendix D) defines two prior distributions: Lie algebra
coefficients (`L`) and matrix composition (`M`).

| Target group | Hypothesis | Command |
|---|---|---|
| C4 (arrow) | SO(2) | `python experiments/flow_matching_2d.py dataset=C4_arrow model=flow_matching/SO2_to_C4_arrow` |
| C4 (arrow) | GL(2,R)+ **(L)** | `python experiments/flow_matching_2d.py dataset=C4_arrow model=flow_matching/GL2_to_C4_arrow_L` |
| C4 (arrow) | GL(2,R)+ **(M)** | `python experiments/flow_matching_2d.py dataset=C4_arrow model=flow_matching/GL2_to_C4_arrow_M` |

### 4.2 3D irregular tetrahedron

For the 3D datasets, use `flow_matching_3d.py` as the `EXP_FILE` with a
different `MODEL` for each hypothesis group. `SO(2)(a)` denotes rotations around
the `z`-axis and `SO(2)(b)` denotes rotations around the tilted axis
`(0, 1/2, -sqrt(3)/2)`.

| Target | Hypothesis | Command |
|---|---|---|
| Tet | SO(3) | `python experiments/flow_matching_3d.py dataset=Tet_irreg_tet model=flow_matching/SO3_irreg_tet_time_power_dist` |
| Oct | SO(3) | `python experiments/flow_matching_3d.py dataset=Oct_irreg_tet model=flow_matching/SO3_irreg_tet_time_power_dist` |
| Ico | SO(3) | `python experiments/flow_matching_3d.py dataset=Ico_irreg_tet model=flow_matching/SO3_irreg_tet_time_power_dist` |
| SO(2)(a) | SO(3) | `python experiments/flow_matching_3d.py dataset=SO2_irreg_tet model=flow_matching/SO3_irreg_tet` |
| SO(2)(b) | SO(3) | `python experiments/flow_matching_3d.py dataset=SO2_b_irreg_tet model=flow_matching/SO3_irreg_tet` |

### 4.3 ModelNet10 with sampled rotations

For ModelNet10, use the `*_modelnet10` datasets. The discrete targets use the
power time-sampling model (`SO3_modelnet10_transformer_time_power`), while the
`SO(2)` targets use the plain transformer (`SO3_modelnet10_transformer`); the
`SO(2)(a)` / `SO(2)(b)` axis conventions are as in §4.2.

| Target | Hypothesis | Command |
|---|---|---|
| Tet | SO(3) | `python experiments/flow_matching_3d.py dataset=Tet_modelnet10 model=flow_matching/SO3_modelnet10_transformer_time_power` |
| Oct | SO(3) | `python experiments/flow_matching_3d.py dataset=Oct_modelnet10 model=flow_matching/SO3_modelnet10_transformer_time_power` |
| Ico | SO(3) | `python experiments/flow_matching_3d.py dataset=Ico_modelnet10 model=flow_matching/SO3_modelnet10_transformer_time_power` |
| SO(2)(a) | SO(3) | `python experiments/flow_matching_3d.py dataset=SO2_modelnet10 model=flow_matching/SO3_modelnet10_transformer` |
| SO(2)(b) | SO(3) | `python experiments/flow_matching_3d.py dataset=SO2_b_modelnet10 model=flow_matching/SO3_modelnet10_transformer` |

### 4.4 Robustness Analysis

Starting from the ModelNet10 (Ico) setup, each sampled point cloud is corrupted
after the true group transformation: `n_masked_points` randomly chosen points
are zeroed out, and a small `SO(3)` jitter `R_noise` near the identity is applied to the coordinates. The jitter comes from the Lie-algebra exponential map, `R_noise = expm(skew(ω))` with `ω ~ N(0, σ² I₃)`, where `σ = dataset.noise_scale` (radians). `σ = 0` disables the noise.

```bash
python experiments/flow_matching_3d.py \
    dataset=Ico_modelnet10_noisy \
    dataset.noise_scale=0.05 \
    dataset.n_masked_points=6 \
    model=flow_matching/SO3_modelnet10_transformer_time_power
```

### 4.5 Real-world data: MI-Motion skeletons

**MI-Motion** ([Peng et al., 2023](https://mi-motion.github.io/)) is a
real-world motion-capture dataset of 3D pedestrian skeletons. Unlike the
previous experiments, **the symmetry group is not imposed** and has to be
inferred from the data.

Expected group: an approximate `C4` rotational symmetry around the `z`-axis, as
pedestrians primarily move along axis-aligned directions while gravity breaks
the full `SO(3)` symmetry.

**Step 1 — preprocess raw skeletons** (only needs to be done once):

Download the raw dataset following the instructions in the
<https://github.com/xiaogangpeng/SocialTGCN> repo.

```bash
# Place the raw MI-Motion data under data/MI-Motion/
# (S0/, S1/, ..., S4/ folders containing .npy sequence files)
python scripts/extract_mi_motion_skeletons.py
# → writes data/MI-Motion/skeletons_normalized.npy
#         data/MI-Motion/metadata.npz
```

**Step 2 — train LieFlow with the `SO(3)` hypothesis group:**

```bash
python experiments/flow_matching_3d.py \
    dataset=SO3_MI_Motion_skeleton \
    model=flow_matching/SO3_MI_Motion_skeleton
```

## 5. Citation

If you use this code, please cite:

```bibtex
@inproceedings{chen2026lieflow,
  title     = {Discovering Symmetry Groups with Flow Matching},
  author    = {Chen, Yuxuan and Park, Jung Yeon and Eijkelboom, Floor and
               Yang, Jianke and van de Meent, Jan-Willem and
               Wong, Lawson L.S. and Walters, Robin},
  booktitle = {Proceedings of the 43rd International Conference on
               Machine Learning (ICML)},
  year      = {2026}
}
```
