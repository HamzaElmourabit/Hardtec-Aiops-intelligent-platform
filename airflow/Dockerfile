FROM apache/airflow:2.10.5-python3.12

USER airflow

RUN python -m pip install --no-cache-dir \
    "numpy==2.5.2" \
    "pandas==3.0.5" \
    "scipy==1.18.0" \
    "scikit-learn==1.9.0" \
    "joblib==1.5.3" \
    "xgboost==3.0.2" \
    "faiss-cpu==1.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.1+cpu" \
    "sentence-transformers==4.1.0" \
    "duckdb==1.3.2"
