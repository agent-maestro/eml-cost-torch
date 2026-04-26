# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] — 2026-04-26 — `diagnose()` predictive risk profile

### Empirical basis (replicated 2026-04-26 with ViT-B/16 added)

  - **n=387 layers across 3 transformers** (GPT-2 small + BERT-base + ViT-B/16),
    **36 PNE samples** (was 24 in initial E-183 run; replicated cleanly).
  - **fp16_drift**: BH-q = 0.020, r_rb = -0.25, ratio 1.10x.
  - **activation_variance**: BH-q = 1.5e-3, r_rb = +0.35, ratio 0.33x.
  - Controls (ResNet-18, EfficientNet-B0) returned all-N/A as expected
    (0 PNE layers each — ReLU and SiLU are EML-elementary).
  - Per-architecture activation_variance survives independently:
    GPT-2 q=8.6e-4, BERT q=2.7e-5, ViT q=2.7e-4.
  - All 24 → 36 PNE samples remain GELU variants. **GELU is currently
    the only modern activation classified as Pfaffian-not-EML** in the
    eml-cost-torch registry (SiLU = x·sigmoid(x) is EML-elementary;
    Mish, GeGLU likewise EML).

### Added

- **`eml_cost_torch.diagnose(model)`** — empirically-grounded per-layer
  risk diagnostic. Returns a `DiagnosisReport` with per-layer
  `LayerRisk` predictions:
    - `fp16_risk` ∈ {`low`, `elevated`}
    - `activation_variance_class` ∈ {`normal`, `saturating`}
  Predictions apply **only** to the two effects that survived BH-FDR
  in the E-183 architecture-diagnostic study (n=275 layers across
  GPT-2 small + BERT-base):
    - **Pfaffian-not-EML layers show ~14% higher fp16 drift** under
      cast (BH-q = 0.022)
    - **Pfaffian-not-EML layers show ~53% lower activation variance**
      under input perturbation (BH-q = 2.1×10⁻⁴)
  The function does NOT run the model — predictions derive from the
  symbolic Pfaffian classification of each layer's class.

- `DiagnosisReport.to_dict()` for JSON serialization. The
  `empirical_basis` is embedded so consumers can audit the
  data backing each prediction.

### Tests

- 8 new in `tests/test_diagnose.py`. Full suite: **35 passing.**

### Mechanistic explanation (a priori, supports the data)

GELU/SiLU/Mish saturate at input tails → bounded outputs (lower
variance) + small derivatives in saturating regions (snap to zero
under fp16 quantization → drift). Same mechanism predicted to apply
to other Pfaffian-not-EML activation classes. Only GELU was
empirically tested in the E-183 corpus — generalization to other
PNE classes pending E-183 expansion.

### Honest caveats

  - All 24 PNE samples in the E-183 corpus are GELU variants. Other
    PNE classes (SiLU, Mish, GeGLU) are theoretically covered by the
    same mechanism but empirically untested in 0.4.0.
  - Drift measurement uses CPU-emulated fp16 cast, not hardware GPU
    fp16. Real-GPU validation pending.
  - The `gradient_norm` hypothesis from the E-183 study is N/A
    (PNE layers have no parameters); not refuted, just untested.
