# Interpretable Machine Learning Models for Heart Disease Prediction

A machine learning pipeline for predicting heart disease using the UCI Cleveland clinical dataset, with 5-fold cross-validation, SHAP interpretability, and an optional LLM feature-extraction proof of concept.

## Project Files
* **Notebook:** `Heart_Disease_Models.ipynb` - Contains the full data preprocessing, model training, evaluation, and interpretability pipeline.

## Dataset
This study utilizes the **UCI Cleveland Heart Disease Dataset**. It contains 303 patient records and 13 clinical features associated with heart health (e.g., age, sex, chest pain type, resting blood pressure, serum cholesterol, etc.).

## Core Dependencies
To run this pipeline, you will need Python 3 and the following libraries:
* `scikit-learn`
* `torch` (PyTorch)
* `xgboost`
* `shap`
* `pandas`
* `numpy`
* `matplotlib`

## How to Run
1. Clone this repository to your local machine or open it in Google Colab/Jupyter Notebook.
2. Ensure all dependencies listed above are installed in your environment.
3. Open `Heart_Disease_Models.ipynb`.
4. Run the cells sequentially from top to bottom. Random seeds have been fixed (`random_state=42` and `torch.manual_seed(42)`) to ensure reproducibility. 

## Important Note on LLM Extraction
The Gemini API extraction section is included strictly as a limited proof-of-concept for unstructured clinical text processing. It demonstrates extracting 10 structured variables from a single synthetic free-text clinical note. This LLM-extracted row is **not** utilized in the primary predictive modeling or the SHAP explanations shown in this notebook.
