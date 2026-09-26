# Table Previews

`preview.pdf` renders the editable ablation fragments and the separately
recovered short-rollout resource table. All use `booktabs`; no manuscript file
was modified. The component proxy is optional appendix material, explicitly
labeled post-hoc and separate from generated-video results.

## Numerical Provenance

- Generation: the user's completed `keepsake_components_60s_n15/ablation.csv`
  and `contrasts.csv`, and `fvd_contrasts.csv` pasted in attachment
  `f2930334-8391-4245-a353-098c45ed0476/Pasted text.txt`.
- Parameter sweep: `keepsake_sensitivity_cpu_60s_n15/summary.csv`, pasted in
  attachment `91014678-cb96-4c04-bc35-9167cfc4b8af/Pasted text.txt`.
- Component proxy: `keepsake_component_proxy_cpu_60s_n15/summary.csv` pasted
  directly in this conversation.
- Resources: actual downloaded profiles and receipts, recovered by
  `paper/recover_resource_measurements.py`. See the adjacent results directory's
  `provenance.json` and `docs/resource_measurements_recovered.md`.

All displayed differences use **variant minus full/default**. The generation
contrast files originally use full minus variant, so signs are reversed and CI
endpoints exchanged. Generation quality point estimates retain all four metrics.
The generation uncertainty panel shows paired contrasts, not the markedly
upward-shifted marginal FVD percentile intervals. No alternative interval was
invented or re-centered, and no statistical test was recomputed for this layout.
No best-score bolding or significance stars are used. Reference labels alone are
bold. Tables do not claim component necessity, optimal constants or equivalence.

## Rendering

From this directory with a LaTeX installation:

```bash
tectonic preview.tex
pdftoppm -scale-to 1800 -png preview.pdf preview
```

The local render used the already available temporary Tectonic binary and cache;
it did not install or change a Conda environment.
