# When a Feature Looks Too Good to Be True

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Article-CC%20BY%204.0-lightgrey.svg)](LICENSE-TEXT)

**Statistical Foundations of Categorical Feature Engineering for Fraud Detection — from Encoding to Inference**

A category with 100% target rate and five observations is more likely evidence of insufficient data than of a robust predictive signal. A correlation of 0.95 between two features is more likely evidence of shared variation than of redundancy. Both illusions share the same root cause: treating a sample statistic as if it were a population parameter. This article formalises why, and provides a framework for encoding decisions that respects what the sample size permits.

## What you will find here

- **Theory:** MLE for category-level target rates, variance under low support, Bayesian smoothing (Beta-Binomial), and why correlation does not imply redundancy.
- **Experiments:** Four reproducible experiments on a synthetic fraud dataset, each testing a specific theoretical claim.
- **Article:** A complete, publication-ready article connecting theory to practice.

## How to reproduce

```bash
# Clone the repository
git clone https://github.com/<your-username>/categorical-features-fraud.git
cd categorical-features-fraud

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run all experiments and generate figures
python scripts/run_all.py
```

## Project structure

```
categorical-features-fraud/
├── article/          # Full article source, references, notation
├── docs/             # Internal docs: thesis, outline, dataset design
├── src/              # Source modules: data, encoding, models, plotting
├── scripts/          # Experiment scripts + run_all.py
├── notebooks/        # Exploratory notebooks (not final)
├── figures/          # Generated figures (300dpi)
├── data/             # Generated dataset (not tracked)
├── notes/            # Study notes per phase
├── config.yaml       # All parameters (seeds, dataset, encoding, figures)
├── requirements.txt  # Pinned Python dependencies
├── CONTRIBUTING.md   # Branch, commit, and merge conventions
└── CHANGELOG.md      # Version history
```

## License

- **Code** (`src/`, `scripts/`): [MIT License](LICENSE)
- **Article text** (`article/`): [CC BY 4.0](LICENSE-TEXT)
