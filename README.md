# Real Estate Buyer Segmentation and Investment Profiling

Machine-learning exploration of real-estate buyer patterns using preprocessing, K-Means and hierarchical clustering, plus an interactive Streamlit dashboard.

> **Data limitation:** The supplied buyer dataset is synthetic practice data, not actual Parcl customer data. Results are illustrative only.

## Files
- `app.py` — interactive Streamlit dashboard
- `analysis.py` — reproducible clustering pipeline
- `requirements.txt` — dependencies
- Add the CSV and executed notebook from the submission ZIP to the repository root before running.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload the project CSV in the dashboard sidebar, or place it beside `app.py`.

Run the analysis:
```bash
python analysis.py
```

## Method
Remove exact duplicate rows and duplicate client IDs; exclude records without region to align with the 594-record analysis set; derive age from date of birth; impute missing values; scale numeric variables and one-hot encode categorical variables. Compare K-Means values k=2 through k=7 using inertia and silhouette score, then profile the resulting buyer groups.

The verified notebook reports a best silhouette score of **0.112616**, indicating weak cluster separation. Treat profiles as exploratory, not as sharply distinct natural segments or investment advice.

Expected CSV filename: `real_estate_buyer_synthetic_dataset.csv`.