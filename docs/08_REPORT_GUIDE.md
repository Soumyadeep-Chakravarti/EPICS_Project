# Report Guide

This document explains the structure of our LaTeX report and what each section contains.

## Report Location

```
report/
├── main.tex           # Main LaTeX document
├── references.bib     # Bibliography (if used)
├── .latexmkrc         # Build configuration
├── Assets/            # Images (logos, biodata photos)
│   ├── VIT_LOGO.png
│   └── ...
├── results/           # Copied visualizations for report
│   ├── confusion_matrix.png
│   └── ...
└── build/             # Compiled output
    └── main.pdf       # Final PDF
```

## Building the Report

### Using latexmk (recommended)

```bash
cd report
latexmk -pdf main.tex
```

### Manual build

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

### Output

The compiled PDF will be at `report/build/main.pdf`

---

## Report Structure

The report follows standard academic format with these sections:

```
1. Title Page
2. Bonafide Certificate
3. Declaration of Originality
4. Acknowledgement
5. Abstract
6. Table of Contents / List of Figures / List of Tables
7. Section 1: Introduction
8. Section 2: Literature Review
9. Section 3: Methodology & Results (Topic of Work)
10. Section 4: Conclusion
11. Section 5: Social Impact
12. Section 6: References
13. Appendix: Publication / Biodata
```

---

## Section-by-Section Guide

### 1. Title Page

**Content**:
- Project title: "Explainable Breast Cancer Detection using Ensemble Learning"
- Team members with registration numbers
- VIT Bhopal University logo
- Degree and date

**When updating**:
- Change team member names/IDs if needed
- Update the date (currently March 2026)

### 2. Bonafide Certificate

**Content**:
- Official certification that this is original work
- Space for supervisor signature
- Space for reviewer signatures

**When updating**:
- Fill in viva date
- Get signatures before submission

### 3. Declaration of Originality

**Content**:
- Statement that work is original
- No plagiarism declaration
- Team member signatures

**When updating**:
- Add date
- Ensure all members sign

### 4. Acknowledgement

**Content**:
- Thanks to supervisor
- Thanks to faculty
- Thanks to team members
- Thanks for open-source tools

**When updating**:
- Add specific supervisor names if known
- Add any additional acknowledgements

### 5. Abstract

**Content**:
- Project summary (1 paragraph)
- Key results (accuracy, ROC-AUC)
- Keywords

**Key points mentioned**:
- 98.25% accuracy (LR & SVM)
- ROC-AUC > 0.99
- 5-fold cross-validation
- Explainability through feature importance

### 6. Table of Contents

**Auto-generated** from section headings.

**Commands used**:
```latex
\tableofcontents
\listoffigures
\listoftables
```

---

## Main Sections

### Section 1: INTRODUCTION

**Subsections**:

| Subsection | Content |
|------------|---------|
| 1.1 Background and Motivation | Why breast cancer detection matters |
| 1.2 Problem Statement | What we're trying to solve |
| 1.3 Scope and Objectives | Specific goals of the project |
| 1.4 Report Organization | Guide to reading the report |

**Key statistics mentioned**:
- 2.3 million new cases annually
- 99% survival rate if caught early
- 29% survival rate if caught late

### Section 2: LITERATURE REVIEW

**Subsections**:

| Subsection | Content |
|------------|---------|
| 2.1 Traditional ML Approaches | SVM, Decision Trees, Random Forest, Logistic Regression |
| 2.2 Ensemble Learning Methods | Bagging, Boosting, Voting, Stacking |
| 2.3 Explainable AI (XAI) | SHAP, LIME, Feature Importance |
| 2.4 Comparative Analysis | Table comparing our work to published papers |

**Key references**:
- Cortes & Vapnik (1995) - SVM
- Breiman (2001) - Random Forest
- Lundberg & Lee (2017) - SHAP
- Recent breast cancer ML papers (2023-2025)

### Section 3: METHODOLOGY & RESULTS

This is the main technical section.

**Subsections**:

| Subsection | Content |
|------------|---------|
| 3.1 System Architecture | Pipeline diagram, data flow |
| 3.2 Algorithm Design | Pseudocode for pipeline and soft voting |
| 3.3 Mathematical Formulations | Metrics equations, ROC-AUC, Gini importance |
| 3.4 Implementation Details | Code snippets, hyperparameters |
| 3.5 Results and Discussion | Performance tables, visualizations |
| 3.6 Sensitivity Analysis | How hyperparameters affect results |
| 3.7 Individual Contributions | Who did what |

**Key figures**:
- System architecture diagram (TikZ)
- Confusion matrix
- ROC curves
- Cross-validation scores
- Feature importance
- Model comparison

**Key tables**:
- Dataset characteristics
- Model hyperparameters
- Performance metrics
- Cross-validation results
- Comparison with published works

### Section 4: CONCLUSION

**Subsections**:

| Subsection | Content |
|------------|---------|
| 4.1 Summary of Achievements | What we accomplished |
| 4.2 Limitations | What could be improved |
| 4.3 Future Work | Next steps |

**Key achievements listed**:
- 98.25% accuracy
- Robust cross-validation (std < 2.5%)
- ROC-AUC > 0.99
- High recall (98.61%)
- Feature importance analysis
- Production-ready saved models

### Section 5: SOCIAL IMPACT

**Subsections**:

| Subsection | Content |
|------------|---------|
| 5.1 Healthcare Accessibility | How ML helps underserved areas |
| 5.2 Educational Impact | Learning value of the project |
| 5.3 Ethical Considerations | Human-in-the-loop, transparency |
| 5.4 SDG Alignment | UN Sustainable Development Goals |

**SDGs mentioned**:
- SDG 3: Good Health
- SDG 4: Quality Education
- SDG 10: Reduced Inequalities

### Section 6: REFERENCES

**Format**: IEEE style

**Categories**:
- Healthcare organizations (WHO, ACS)
- Datasets (UCI WDBC)
- Foundational ML papers
- XAI papers
- Recent breast cancer research
- Software tools (scikit-learn, pandas)

---

## LaTeX Tips

### Adding a Figure

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{results/confusion_matrix.png}
\caption{Confusion Matrix of Ensemble Model}
\label{fig:confusion}
\end{figure}
```

Reference it with: `Figure \ref{fig:confusion}`

### Adding a Table

```latex
\begin{table}[H]
\centering
\caption{Model Performance Metrics}
\label{tab:results}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} \\ \hline
Logistic Regression & 0.9825 & 0.9861 & 0.9861 \\ \hline
\end{tabular}
\end{table}
```

Reference it with: `Table \ref{tab:results}`

### Adding an Equation

```latex
\begin{equation}
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}
```

### Adding Code

```latex
\begin{lstlisting}[style=pythonstyle, caption={Example Code}]
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=models, voting='soft')
\end{lstlisting}
```

### Algorithm Pseudocode

```latex
\begin{algorithm}[H]
\caption{Pipeline Algorithm}
\begin{algorithmic}[1]
\Require Dataset $\mathcal{D}$
\Ensure Trained model
\State Load data
\State Preprocess
\State Train models
\State \Return ensemble
\end{algorithmic}
\end{algorithm}
```

---

## Common Issues

### Problem: Figures not found

**Solution**: Ensure `results/` folder in `report/` contains the images:
```bash
cp ../results/*.png report/results/
```

### Problem: References not showing

**Solution**: Run LaTeX twice:
```bash
pdflatex main.tex
pdflatex main.tex
```

### Problem: Undefined reference errors

**Solution**: Check that all `\ref{}` commands match existing `\label{}` commands.

### Problem: Table too wide

**Solution**: Use `\resizebox` or reduce column count:
```latex
\resizebox{\textwidth}{!}{
\begin{tabular}{...}
```

---

## Customization

### Changing Margins

In the preamble:
```latex
\usepackage[a4paper, margin=1in]{geometry}
```

### Changing Line Spacing

```latex
\setstretch{1.1}  % Currently 1.1
```

### Adding a New Section

```latex
\section{NEW SECTION TITLE}
\subsection{Subsection}
Content goes here.
```

---

## Files to Update Before Submission

1. **Title page**: Verify names and date
2. **Certificate**: Fill in viva date
3. **Declaration**: Add submission date
4. **Acknowledgement**: Add supervisor name
5. **Abstract**: Verify results match code output
6. **Results section**: Update if you re-run experiments
7. **Individual contributions**: Verify accuracy
8. **Biodata section**: Update if needed

---

## Quick Reference

| Element | Command |
|---------|---------|
| Bold | `\textbf{text}` |
| Italic | `\textit{text}` |
| Section | `\section{Title}` |
| Subsection | `\subsection{Title}` |
| Bullet list | `\begin{itemize}...\end{itemize}` |
| Numbered list | `\begin{enumerate}...\end{enumerate}` |
| Figure | `\includegraphics[width=0.7\textwidth]{path}` |
| Table | `\begin{tabular}...\end{tabular}` |
| Equation | `\begin{equation}...\end{equation}` |
| Reference | `\ref{label}` |
| Citation | `\cite{key}` |

---

**Next**: Check the [Glossary](09_GLOSSARY.md) for quick term definitions.
