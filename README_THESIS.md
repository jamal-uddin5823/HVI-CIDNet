# Bachelor's Thesis Report

## Title
**Face Recognition-Aware Low-Light Image Enhancement using Discriminative Loss Functions**

## Files

- `thesis_report.tex` - Main LaTeX document
- `references.bib` - Bibliography in BibTeX format
- `figures/` - Directory for figures (needs to be populated)

## Compilation

### Using pdflatex (recommended)
```bash
pdflatex thesis_report.tex
bibtex thesis_report
pdflatex thesis_report.tex
pdflatex thesis_report.tex
```

### Using latexmk
```bash
latexmk -pdf thesis_report.tex
```

## Required Figures

The following figures need to be placed in the `figures/` directory:

1. `pipeline.png` - Overview of the proposed face recognition-aware low-light enhancement pipeline
2. `ablation.png` - EER vs face loss weight graph

## Figure Generation Scripts

To generate the required figures from your results:

```bash
# Generate ablation curve
python scripts/generate_ablation_curve.py --results-dir results/discriminative

# Generate pipeline diagram
python scripts/generate_pipeline_diagram.py
```

## Report Structure

1. **Abstract** - Summary of the thesis
2. **Background** - Low-light enhancement, HVI-CIDNet, face verification metrics
3. **Proposed Methodologies** - Physics-based synthesis, multi-level dataset, discriminative face loss
4. **Implementation** - System architecture, training configuration, datasets
5. **Results and Analysis** - Main results, ablation study, discussion
6. **Conclusion** - Summary, limitations, future work
7. **References** - IEEE-style bibliography

## Page Count

- Abstract: ~1 page
- Background: ~3-4 pages
- Methodologies: ~5-6 pages
- Implementation: ~3-4 pages
- Results: ~4-6 pages
- Conclusion: ~2-3 pages
- **Total: ~18-24 pages**

## Dependencies

- TeX Live or MiKTeX
- LaTeX packages: graphicx, amsmath, booktabs, hyperref, cite, float, caption, subcaption, algorithm, algorithmic

## Notes

- The report uses IEEE citation style
- All equations are numbered
- Tables use booktabs for professional formatting
- Figures are referenced with Figure~\ref{...} syntax
