#  Energy — Data Scientist Demo Project

This demo project showcases a compact, interview-friendly Python project tailored to the RABOT Energy Data Scientist role.
It includes:
- data ingestion & preprocessing (sample CSVs)
- a simple forecasting pipeline (scikit-learn RandomForest)
- a charging optimization example (pulp linear program) that schedules EV charging to minimize cost
- a small A/B test simulation with statistical test (scipy)
- plotting utilities and a single `run_demo.py` CLI for quick runs

## Quick start (local)
1. Create a virtualenv and install requirements:
```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```



## Streamlit App

A Streamlit app is included (`streamlit_app.py`) that provides an interactive interface to:
- Run the forecasting pipeline
- Run the EV charging optimization and download schedules
- Simulate an A/B test and download results

Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
