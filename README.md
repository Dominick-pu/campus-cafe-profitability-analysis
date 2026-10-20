# Campus Café Profitability Analysis

## Project Overview

This Python project analyzes café sales, forecasts product demand, estimates price elasticity, and evaluates pricing and promotional strategies to support data-driven profitability decisions.

## Data and Confidentiality

To protect business confidentiality, this public repository uses **synthetic sales, product, and inventory data** rather than disclosing confidential operational records.

The Python code can automatically generate all synthetic datasets. The CSV files are also uploaded solely to facilitate data inspection, result verification, and reproducibility. They are not required for the program to run.

## Methodology

- **Data Processing:** Clean and preprocess POS-style transaction data using Pandas and NumPy.
- **Predictive Modeling:** Compare Linear Regression, Ridge Regression, and Random Forest to forecast product demand.
- **Price Elasticity:** Estimate demand sensitivity and identify revenue-maximizing price points.
- **Profitability Analysis:** Calculate gross margins, inventory turnover, and identify slow-moving products and high-margin combinations.
- **Optimization:** Simulate pricing adjustments and promotional strategies to evaluate potential gross profit improvements.
- **Visualization:** Use Matplotlib to illustrate demand patterns, price elasticity, and revenue–profit relationships.

## Key Results

Using 180 days of synthetic data across six products:

- Estimated product-level demand elasticities.
- Evaluated forecasting models using chronological train, validation, and test splits.
- Identified pricing and promotional opportunities.
- Achieved a **6.24% simulated increase in predicted gross profit**, from approximately $112,427 to $119,444.

## How to Run

Install dependencies:

`pip install -r requirements.txt`

Run the analysis:

`python coffee_profitability_analysis.py`

If the input CSV files are absent, the program automatically generates synthetic data. To regenerate the datasets:

`python coffee_profitability_analysis.py --generate-demo`

Results and visualizations are saved in the `outputs/` folder.

## Limitations

All publicly reported results are based on synthetic data and model assumptions. Predicted gross profit improvements are illustrative and do not represent verified real-world business outcomes.
