# Interpretable Machine Learning for Retrospective Coronary Artery Disease Classification Using the UCI Cleveland Dataset

A leak-free, interpretable machine learning pipeline for retrospective coronary artery disease (CAD) classification using the 303-patient UCI Cleveland Heart Disease dataset. The repository includes stratified 5-fold cross-validation, nested 5×3 cross-validation (for classical models), 10-seed repeated holdout evaluation, Nadeau–Bengio variance-corrected pairwise significance testing, progressive 4-stage clinical workup analysis, unified pipeline-level Permutation SHAP explainability, and a limited LLM feature-extraction proof of concept.

## Project Files

- **`Heart_Disease_Models.ipynb`**: Contains the complete workflow, including programmatic dataset loading, clinical feature audit, leak-free preprocessing, model training and evaluation, statistical testing, staged clinical feature ablation, unified Permutation SHAP analysis, and the optional LLM extraction demonstration.

## Dataset and Clinical Framing

This study uses the **UCI Cleveland Heart Disease Dataset** (loaded via `ucimlrepo`, ID `45`), comprising **303 patient records** (`164` non-CAD, `139` angiography-confirmed CAD) and **13 clinical variables** organized by clinical role:

- **Independent Cardiovascular Risk Factors (5):** `age`, `sex`, `trestbps`, `chol`, `fbs`
- **Clinical Symptom / Presentation (1):** `cp` (chest pain type)
- **Baseline Diagnostic Measurement (1):** `restecg` (resting ECG)
- **Exercise Stress ECG Measurements (4):** `thalach`, `exang` (exercise-induced angina), `oldpeak`, `slope`
- **Nuclear Imaging Diagnostic (1):** `thal` (thallium-201 stress myocardial perfusion scintigraphy: normal, fixed defect, reversible defect)
- **Invasive Angiographic Diagnostic (1):** `ca` (number of major vessels colored by fluoroscopy)

The target clinical point in care is retrospective decision support during a specialty outpatient cardiology evaluation of symptomatic patients after exercise stress testing and thallium-201 perfusion imaging, prior to invasive coronary angiography (**Stage 3, 12 features**). Models are evaluated across four nested clinical workup stages (Stages 1–4), with the full 13-feature model including `ca` (**Stage 4**) serving as a partially circular retrospective benchmark.

## Models and Validation Protocol

Five classifiers are evaluated: **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**, and a **PyTorch Multilayer Perceptron (MLP)** (`128–64–32` hidden units with 1D BatchNorm, LeakyReLU, Dropout `0.4`, and AdamW).

- **Locked Holdout Split:** The dataset is partitioned via a locked 80:20 stratified split (`random_state=42`) into a 242-patient development set and a 61-patient holdout test set. All imputers, scalers, and one-hot encoders are fit strictly inside each training split.
- **Cross-Validation and Stability Checks:** Primary evaluation uses leak-free 5-fold cross-validation on the 242-patient development set, alongside 10-seed repeated 80:20 holdout splits (`seeds 42–51`) and 5×3 nested cross-validation (`inner 3-fold GridSearchCV`) for the four classical models.
- **Note on MLP Hyperparameter Selection:** The classical models did not use the 61 holdout patients during hyperparameter selection. The PyTorch MLP architecture and regularization settings were chosen during initial exploration on the `seed=42` split, so its single-split holdout metrics do not independently confirm the cross-validation results. During leak-free MLP training, epoch checkpointing relies solely on an internal 85:15 stratified train/validation split (`n=205` / `n=37`) without monitoring the holdout set.
- **Statistical Testing:** Pairwise comparisons across the 5 development CV folds use the Nadeau and Bengio (2003) variance-corrected paired *t*-test (`n_test / n_train = 0.25`, `df = 4`) with population standard deviations (`ddof = 0`) and 95% confidence intervals.
- **Explainability:** Global and local feature attributions are computed for both Random Forest and the MLP using a unified pipeline-level `shap.PermutationExplainer` on the 13 original clinical variables prior to one-hot encoding, using a 100-patient stratified training background set.

## Core Dependencies

To run this pipeline, you will need **Python 3** and the following packages:

- `ucimlrepo`
- `scikit-learn`
- `torch` (PyTorch)
- `xgboost`
- `shap`
- `scipy`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## How to Run

1. Clone this repository or open `Heart_Disease_Models.ipynb` directly in Google Colab / Jupyter Notebook.
2. Ensure all dependencies listed above are installed in your environment (the notebook installs `ucimlrepo` in Cell 1).
3. Run the cells sequentially from top to bottom.
4. Random seeds are fixed (`random_state=42`, `np.random.seed(42)`, and `torch.manual_seed(42)`) to reproduce the reported metrics and figures.

## Important Note on LLM Feature Extraction

The Gemini API extraction cell (`gemini-1.5-flash`) is included strictly as a limited proof of concept for clinical text processing. It demonstrates mapping a single synthetic free-text clinical note into 10 structured variables (excluding `slope`, `ca`, and `thal`). This LLM-extracted record is not used in model training, evaluation, or SHAP explainability.
