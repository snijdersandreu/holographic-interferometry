# Photonics Presentation: Holographic Interferometry

This repository contains the LaTeX source for a presentation on Holographic Interferometry.

## Compilation

To compile the presentation, you will need a LaTeX distribution installed (e.g., TeX Live, MiKTeX, or MacTeX).

Navigate to the project directory in your terminal and run the following command twice to resolve references and create the PDF:

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

This will generate a `presentation.pdf` file.

## Structure

- `presentation.tex`: The main LaTeX file for the presentation.
- `references.bib` (optional): For managing bibliography with BibTeX. If you use this, you'll also need to run `bibtex presentation` and then `pdflatex` twice more.

## Customization

- **Theme**: You can change the Beamer theme by modifying the `\usetheme{}` command in `presentation.tex`. A list of available themes can be found in the Beamer documentation.
- **Content**: Edit `presentation.tex` to add your specific content, images, and equations.
- **Author**: Remember to replace "Your Name" in `presentation.tex` with your actual name. 