# 4SC000 Quarto framework -- how we'll work together

## What's here

```
course-quarto/
  _quarto.yml               project-wide settings (formats, footer, theme)
  styles/
    macros.tex               shared LaTeX notation (\costtogo, \policy, ...) for PDF output
    macros-mathjax.html       same notation, MathJax version, for browser/revealjs output
    tue.scss                  slide theme (placeholder colors -- tell me if you want real TU/e branding)
  lectures/
    lecture-template.qmd      mechanics demo: sections, math, live code plot, image, summary
  exercises/
    exercise-template.qmd     mechanics demo: problem statement + collapsible solution
```

Both templates are demo/mechanics only, not real course content yet -- see
"Lecture 1 and 2, next" below.

## Prerequisite: installing Quarto

I can't install the Quarto CLI in my own sandbox (its release download is
outside my network allowlist), so I've only been able to sanity-check the
underlying Markdown/math/slide structure with plain `pandoc`, not do a full
Quarto render myself. You'll need Quarto installed locally to actually
render/preview what we build:

- Mac: `brew install quarto`, or download the `.pkg` from quarto.org
- Then: `cd course-quarto && quarto preview lectures/lecture-template.qmd`
  (opens a live-reloading browser preview) or `quarto render lectures/lecture-template.qmd`
  for a one-off HTML file.

If installing feels like friction you don't want, say so -- LaTeX Beamer for
slides + PDF-only exercises is a fallback that avoids a new tool entirely,
at the cost of losing the live-code-in-slides and clickable HTML-solution
features.

## How we'll collaborate

- I edit `.qmd` files as plain text -- same as how we already work with your
  `.tex` problem sets. You can review changes as a diff.
- One file per lecture, one file per problem set (matching your current
  `Lecture1.key`, `PS1.tex` naming, just with `.qmd`).
- Notation: add new macros to `styles/macros.tex` *and*
  `styles/macros-mathjax.html` (both, so PDF and browser slides match) rather
  than hardcoding LaTeX inline in every slide.
- Solutions live inline in the same `.qmd` as the problem, inside a
  `::: {.callout-note collapse="true" title="Solution"}` block -- collapsed
  by default in both the HTML and PDF output, so one file serves as both the
  student handout and your solutions manual.

## Lecture 1 and 2, next

I cannot open `.key` files -- Keynote's format is binary and opaque to me,
so I don't know what's currently on those slides. To do the "correct slides
1 and 2" step, I need one of:

1. Export `Lecture1.key` / `Lecture2.key` to PDF from Keynote
   (File > Export To > PDF) and share the PDFs, or
2. Paste in the outline/bullet text and describe the figures, or
3. I draft lecture 1-2 content fresh from the learning outcomes and PS1
   mapping, and you correct it against what you actually teach.

(1) is fastest and lets me see figures/diagrams too, not just text.

---

## Publishing

`index.qmd` is the site landing page (links to the four forms of each
unit). To put the rendered site online -- GitHub Pages now, a TU/e
address later -- see **PUBLISHING.md**, and then `./publish.sh`.
