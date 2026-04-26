# LaTeX Report for Overleaf

## Files Created

### Main Files
- `main.tex` - Main LaTeX document
- `references.bib` - Bibliography (32 references)

### Sections (in `sections/` directory)
- `literature_review.tex` - Section 2
- `methodology.tex` - Section 3 (includes NLP, LLM, RAG, GNN, Fusion, Explainability)
- `results.tex` - Section 4 (all results tables and analysis)
- `conclusion.tex` - Section 5
- `appendix.tex` - Appendices

## How to Use in Overleaf

### Option 1: Upload Files
1. Create a new project in Overleaf
2. Upload all `.tex` files and `references.bib`
3. Create a `sections/` folder and upload section files there
4. Compile with pdfLaTeX

### Option 2: Copy-Paste
1. Create a new project in Overleaf
2. Replace `main.tex` content with the provided file
3. Create new files for each section
4. Create `references.bib` and paste the bibliography

## Document Structure

```
main.tex
├── Title, Author, Abstract, Keywords
├── Table of Contents
├── Section 1: Introduction (inline in main.tex)
├── Section 2: Literature Review (literature_review.tex)
├── Section 3: Methodology (methodology.tex)
├── Section 4: Results and Discussion (results.tex)
├── Section 5: Conclusion and Future Work (conclusion.tex)
├── References (references.bib)
└── Appendices (appendix.tex)
```

## Formatting

- **Font:** Times New Roman, 12pt
- **Spacing:** 1.5 (onehalfspacing)
- **Margins:** 1 inch all sides
- **Paper:** A4
- **Column:** Single column
- **Bibliography Style:** IEEEtran

## Compilation

The document uses standard LaTeX packages:
- `times` - Times New Roman font
- `setspace` - Line spacing
- `graphicx` - Figures (if you add them)
- `amsmath`, `amssymb` - Math symbols
- `algorithm`, `algorithmic` - Algorithms
- `hyperref` - Clickable links
- `cite` - Citations
- `booktabs` - Professional tables
- `listings` - Code listings

## Adding Figures

To add the architecture diagram or other figures:

1. Upload image files to Overleaf
2. Use this code:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{architecture.png}
\caption{System Architecture}
\label{fig:architecture}
\end{figure}
```

## Page Count

Expected output: **35-40 pages** when compiled

## Statistics

- **Word Count:** ~6,300 words
- **References:** 32 citations
- **Tables:** 6 results tables
- **Sections:** 5 main sections + appendices

## Customization

### Change Author Information
Edit lines 28-34 in `main.tex`:
```latex
\author{
    Your Name \\
    \textit{Your Institution} \\
    ...
}
```

### Change Title
Edit line 26 in `main.tex`

### Add More References
Add entries to `references.bib` and cite with `\cite{key}`

## Compilation Steps in Overleaf

1. Click "Recompile" button
2. If bibliography doesn't appear:
   - Menu → Settings → Compiler → pdfLaTeX
   - Recompile 2-3 times (for references to resolve)

## Common Issues

**Bibliography not showing:**
- Compile 2-3 times
- Check that `references.bib` is in the root directory

**Section files not found:**
- Ensure `sections/` folder exists
- Check file names match exactly

**Tables too wide:**
- Reduce font size: `\small` or `\footnotesize` before table
- Rotate table: use `\begin{sidewaystable}` (requires `rotating` package)

## Final Checklist

- [ ] All sections compile without errors
- [ ] References appear correctly
- [ ] Tables are properly formatted
- [ ] Page numbers are present
- [ ] Table of contents is generated
- [ ] Abstract is on first page
- [ ] All citations resolve
- [ ] No overfull hbox warnings (or minimal)

## Download PDF

Once compiled in Overleaf:
1. Click "Download PDF" button
2. Submit the PDF file

## Notes

- All results are based on actual training runs
- Tables contain real performance metrics
- References include foundational and recent papers
- Code examples match your implementation
- Ethical considerations are included

