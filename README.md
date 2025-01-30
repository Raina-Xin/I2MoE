# I<sup>2</sup>MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
Code for *I<sup>2</sup>MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts*.

## Data Directory

Create data directory under `./data`

## Train Models

### Train I<sup>2</sup>MoE models

- Supported fusion methods: `<fusion>` in `transformer`, `interpretcc`, `moepp`, `switchgate`.
- Supported datasets:`<dataset>` in `adni`, `mimic`, `mmimdb`, `mosi_regression`, `enrico`.

```
source scripts/train_scripts/imoe/<fusion>/run_<dataset>.sh
```

### Train vanilla fusion models

- Supported fusion methods: `<fusion>` in `transformer`, `interpretcc`, `moepp`, `switchgate` and other fusion (`flexmoe`, `lrtf`, `m3care`, `shaspec`).
- Supported datasets:`<dataset>` in `adni`, `mimic`, `mmimdb`, `mosi_regression`, `enrico`.

```
# For <fusion> in ["transformer", "interpretcc", "moepp", "switchgate"]
source scripts/train_scripts/baseline/<fusion>/run_<dataset>.sh
# For <fusion> in ["flexmoe", "lrtf", "m3care", "shaspec"]
source scripts/train_scripts/baseline/other_fusion/run_<dataset>.sh

```

## Ablations of I<sup>2</sup>MoE

- Supported datasets:`<dataset>` in `adni`, `mimic`, `mmimdb`, `mosi_regression`, `enrico`.

```
source scripts/train_scripts/latent_contrastive/transformer/run_<dataset>.sh
source scripts/train_scripts/less_perturbed_forward/transformer/run_<dataset>.sh
source scripts/train_scripts/synergy_redundancy_only/transformer/run_<dataset>.sh
source scripts/train_scripts/simple_weighted_average/transformer/run_<dataset>.sh
source scripts/train_scripts/no_interaction_loss/transformer/run_<dataset>.sh
```
