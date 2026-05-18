# AI-Assisted Collaborative Research Disclosure

This document discloses the role of AI assistance in the development of the
research and artifacts contained in this repository and the accompanying paper:

> **Locally Conservative, Globally Not: Diagnosing the Lagrangian Structure
> of Pretrained Transformers**
> Dimitar P. Gueorguiev (Independent Researcher), 2026.

---

## Overview

This work was developed through a human–AI collaborative workflow using
**Anthropic's Claude** (Opus and Sonnet model families) as a research tool
throughout the project lifecycle. The collaboration is consistent with TMLR's
LLM use policy, which permits LLMs as general-purpose assistive tools provided
the author takes full responsibility for the content.

---

## Domains of AI Contribution

The AI system contributed in the following areas:

### Mathematical Derivation Assistance
- Algebraic manipulation and verification of the STP–acceleration identity
- Cross-checking of proof steps and notation consistency

### Experiment Implementation and Debugging
- Shared-potential regression scripts (`shared_potential_fit.py`,
  `splm_oracle_fit.py`, `token_direction_fit.py`, `sharedV_capacity_sweep.py`)
- Trajectory extraction pipeline and model-variant auto-detection
- Colab orchestration notebooks for GPU execution
- Debugging of checkpoint-loading issues across SPLM variants

### Result Analysis
- Interpretation of per-layer R² profiles and architectural separator margins
- Oracle R² interpretation under LayerNorm non-linearity
- PCA-dimension robustness analysis (PCA-16 vs PCA-32 sweep)

### Paper Drafting and Technical Exposition
- First drafts of paper sections, subsequently reviewed and revised by the author
- LaTeX typesetting, cross-referencing, and notation consistency
- Prose editing and literature-style expansion

### Repository Organisation
- README structure, provenance JSON system, and documentation

---

## What Remained Under Human Control

- **Research direction and conceptual framing** — the decision to pursue the
  STP–acceleration identity and the shared-potential separator as a standalone
  contribution originated with the author
- **Hypothesis selection** — which architectures to compare, which diagnostic
  to develop, which validity controls to run
- **Verification of all results** — every mathematical claim and every
  experimental number was checked by the author
- **Experimental supervision** — training runs, hyperparameter decisions,
  convergence monitoring, checkpoint selection
- **Final scientific judgment** — the author understands and can defend all
  claims made in the paper and takes full responsibility for its contents

---

## On Authorship

This work is authored solely by Dimitar P. Gueorguiev. Claude is acknowledged
as a collaborative research tool, not as a co-author, consistent with TMLR's
policy that LLMs are not eligible for authorship.

---

*This disclosure was last updated on May 18, 2026.*
