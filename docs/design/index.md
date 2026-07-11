# Design documents

The original project design documents, included **verbatim** (the
committed Markdown files are byte-identical to the source documents; only
the math delimiters are converted at documentation build time so the
formulas render).

The parent proposal defines the locked seven-step structure; the
step-specific notes expand Steps 1, 3, 4, 5 and 6. Steps 0, 2 and 7 are
specified inside the parent proposal itself.

```{toctree}
:maxdepth: 1

madmax_closed_loop_calibration_proposal
madmax_step1_booster_proposal_technical_design
madmax_step3_antenna_alignment_technical_design
madmax_step4_measurement_technical_design
madmax_step5_joint_inference_technical_design
madmax_step6_predictive_model_technical_design
```

## Implementation decisions

Where the design notes defer choices to the MADMAX team, this
implementation had to pick concrete defaults. They are catalogued — with
the items needing experimental-team review marked ⚠ — in:

```{toctree}
:maxdepth: 1

DESIGN_DECISIONS
```

For the mapping from design sections to code, see
{doc}`../user_guide/architecture` and the per-step pages under
{doc}`../algorithm/index`.
