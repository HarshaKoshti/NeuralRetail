# NeuralRetail – AI Sales Intelligence

AI-Powered Retail Analytics Platform built for Amdox Technologies

## Live Demo
[Click Here](https://neuralretail-dtkwv5r5zdrh7p2ifjrdw.streamlit.app)

## Models
| Model | Algorithm | Metric | Result |
|-------|-----------|--------|--------|
| Demand Forecast | Prophet | MAPE | 41.16% |
| Churn Prediction | XGBoost | AUC-ROC | 1.0000 |
| Segmentation | K-Means | Silhouette | 0.55+ |

## Tech Stack
- Python, Pandas, Prophet, XGBoost, SHAP
- Streamlit, Plotly, Scikit-learn
- GitHub, Streamlit Cloud

## Setup
pip install -r requirements.txt
streamlit run Dashboard/app.py

## Project Structure
NeuralRetail/
├── Dashboard/app.py      # Streamlit dashboard
├── Notebook/             # Jupyter notebooks
├── run_all.py            # Run all models
└── requirements.txt

## Author
Harsha Koshti | Amdox Technologies | April 2026
