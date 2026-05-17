# `figures/` — figures rendered for the paper

This directory holds the PNG / PDF figures that the paper embeds. The
figures *are* tracked in git so that a clean clone of this repository
contains everything the paper needs.

| File | Paper section | Producer |
|---|---|---|
| `stp_acceleration_equivalence.png` | §4 Fig. 1 | [`../notebooks/stp_loss/energy_landscape_validation.ipynb`](../notebooks/stp_loss/energy_landscape_validation.ipynb) |
| `tangential_acceleration.png`      | §5 Fig. 2 | [`../notebooks/stp_loss/energy_landscape_validation.ipynb`](../notebooks/stp_loss/energy_landscape_validation.ipynb) |
| `trajectory_acceleration_profiles.png` | §5 Fig. 3 | [`../notebooks/stp_loss/energy_landscape_validation.ipynb`](../notebooks/stp_loss/energy_landscape_validation.ipynb) |
| `pythia_replication_panel.png`     | §6 Fig. 4 | [`../notebooks/cross_model/pythia_tangential_acceleration.ipynb`](../notebooks/cross_model/pythia_tangential_acceleration.ipynb) |
| `three_way_separator_panel.png`    | §8 Fig. 5 | [`../notebooks/conservative_arch/plot_three_way_comparison.py`](../notebooks/conservative_arch/plot_three_way_comparison.py) |
| `per_layer_R2_profile.png`         | §8.1 / 8.3 / 8.4 Fig. 6 | [`../notebooks/conservative_arch/plot_sharedV_comparison.py`](../notebooks/conservative_arch/plot_sharedV_comparison.py) |
| `oracle_fit_table.png`             | §9.1 Fig. 7 | [`../notebooks/conservative_arch/splm_oracle_fit.py`](../notebooks/conservative_arch/splm_oracle_fit.py) |
| `capacity_sweep_curves.png`        | §9.2 Fig. 8 | [`../notebooks/conservative_arch/sharedV_capacity_sweep.py`](../notebooks/conservative_arch/sharedV_capacity_sweep.py) |
| `token_direction_replication.png`  | §9.3 Fig. 9 | [`../notebooks/conservative_arch/token_direction_fit.py`](../notebooks/conservative_arch/token_direction_fit.py) |
| `jacobian_symmetry_pca16.png`      | §7.2 Fig. 10 | [`../notebooks/conservative_arch/jacobian_symmetry.py`](../notebooks/conservative_arch/jacobian_symmetry.py) |

To regenerate all figures from scratch:

```bash
# §4 / §5
cd ../notebooks/stp_loss/ && \
  jupyter nbconvert --to notebook --execute --inplace energy_landscape_validation.ipynb

# §6
cd ../cross_model/ && \
  jupyter nbconvert --to notebook --execute --inplace pythia_tangential_acceleration.ipynb

# §7 / §8 / §9 (assumes the SPLM positive-control and matched-baseline
# checkpoints are present in ../../checkpoints/; see checkpoints/README.md
# for how to produce them)
cd ../conservative_arch/ && \
  python jacobian_symmetry.py        && \
  python run_full_pipeline.py        && \
  python plot_three_way_comparison.py && \
  python plot_sharedV_comparison.py   && \
  python splm_oracle_fit.py           && \
  python sharedV_capacity_sweep.py    && \
  python token_direction_fit.py
```
