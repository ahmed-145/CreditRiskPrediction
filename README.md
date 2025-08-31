# Credit Risk Prediction

A machine learning-based credit risk assessment system that predicts loan default probability for banking applications. Developed during an internship at NBK Egypt using real banking data from Oracle databases.

## 🎯 Project Overview

This project implements a comprehensive credit risk prediction pipeline that:
- Extracts and preprocesses banking data from Oracle databases (1000+ tables)
- Engineers financial features and ratios for loan assessment
- Trains machine learning models to predict loan repayment likelihood
- Provides an interactive Streamlit web interface for real-time predictions

## 🏗️ Architecture

### Data Pipeline
1. **Data Extraction**: Oracle Developer App → 1000+ tables (Tememnos R22 AA)
2. **Feature Engineering**: Calculate 17+ derived financial ratios
3. **Preprocessing**: Encoding, scaling, PCA dimensionality reduction
4. **Model Training**: Random Forest with SMOTE balancing
5. **Deployment**: Streamlit UI with real-time prediction

### Technology Stack
- **Backend**: Python, scikit-learn, pandas, numpy
- **UI**: Streamlit
- **Database**: Oracle (Tememnos R22 AA banking system)
- **ML Pipeline**: RandomForest, PCA, SMOTE, StandardScaler
- **DevOps**: Docker, OLLAMA, OpenWebUI (experimental)

## 📊 Features

### Core Financial Metrics
- **Debt-to-Income Ratio (DTI)**
- **Monthly Payment Burden**
- **Interest Coverage Ratio**
- **Principal-to-Interest Ratio**
- **Effective Interest Rate**
- **Loan-to-Income Exposure**

### Input Categories
- 🌍 **Geographical**: Residence, nationality, birthplace
- 👤 **Demographics**: Age, gender, marital status, income
- 🏦 **Account Info**: Customer type, account history, contact dates
- ⚠️ **KYC/Risk**: Compliance status, risk categorization
- 💰 **Loan Details**: Amount, duration, interest rate, type
- 💳 **Charges**: Fee structures, posting restrictions

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn
```

### Training the Model
```bash
python train_and_save_artifacts.py
```
This creates the required artifacts:
- `model.pkl` - Trained RandomForest classifier
- `scaler.pkl` - StandardScaler for feature normalization
- `pca.pkl` - PCA transformer for dimensionality reduction
- `features.pkl` - Feature list in training order
- `defaults.pkl` - Median values for missing data imputation
- `encoders.pkl` - Label encoders for categorical variables
- `metadata.pkl` - Date columns and categorical column metadata

### Running the Application
```bash
streamlit run app.py
```

## 🎛️ Usage Modes

### 1. Human-Friendly Input
Interactive form with organized sections:
- Dropdown menus for categorical data
- Date pickers for temporal fields
- Numeric inputs with validation
- Real-time calculation of derived financial metrics

### 2. CSV Row Input
Paste raw CSV data for bulk processing:
- Automatic preprocessing pipeline
- Handles missing values and data type conversion
- Maintains compatibility with training data format

## 🧮 Feature Engineering

The system automatically calculates 17 derived financial features:

```python
# Example derived features
DTI = TOTAL_LOAN_AMOUNT / TOTAL_MONTHLY_INCOME
MONTHLY_PAYMENT_BURDEN = MONTHLY_PAYMENT / TOTAL_MONTHLY_INCOME
INTEREST_COVERAGE = TOTAL_LOAN_AMOUNT / TOT_INTEREST_AMT
PRINCIPAL_TO_INTEREST_RATIO = LOAN_AMOUNT / TOT_INTEREST_AMT
```

## 📈 Model Performance

The RandomForest classifier with SMOTE balancing provides:
- **Target Classes**: 
  - PDO (Past Due/Overdue) = 0 (High Risk)
  - CUR (Current) = 1 (Low Risk)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Class Balancing**: SMOTE oversampling for imbalanced datasets

## 🗂️ File Structure

```
CreditRiskPrediction/
├── train_and_save_artifacts.py  # Model training pipeline
├── app.py                       # Streamlit web application
├── feature eng.csv              # Preprocessed training data
├── model.pkl                    # Trained model artifacts
├── scaler.pkl                   # Feature scaling transformer
├── pca.pkl                      # PCA dimensionality reducer
├── features.pkl                 # Feature list metadata
├── defaults.pkl                 # Default values for imputation
├── encoders.pkl                 # Categorical encoders
├── metadata.pkl                 # Pipeline metadata
└── README.md                    # Project documentation
```

## 🔄 Pipeline Workflow

1. **Data Loading**: Read CSV with automatic date parsing
2. **Feature Engineering**: Calculate financial ratios and derived metrics
3. **Encoding**: Transform categorical variables using LabelEncoder
4. **Date Processing**: Extract year, month, day, and days-since features
5. **Scaling**: Standardize numeric features using StandardScaler
6. **Dimensionality Reduction**: Apply PCA (95% variance retention)
7. **Balancing**: Use SMOTE for minority class oversampling
8. **Training**: Fit RandomForest with balanced class weights
9. **Evaluation**: Comprehensive metrics and confusion matrix analysis

## 💡 Key Innovations

- **Comprehensive Feature Engineering**: 17+ derived financial ratios
- **Robust Date Handling**: Automatic parsing and feature extraction
- **Dual Input Modes**: Human-friendly UI + CSV batch processing
- **Real-time Calculations**: Live financial metric computation
- **Production Ready**: Artifact-based deployment with proper error handling

## 🔍 Technical Details

### Data Preprocessing
- Missing value imputation using median values
- Categorical encoding with fallback handling
- Date feature extraction (year, month, day, days_since)
- Feature scaling and PCA transformation

### Model Architecture
- **Algorithm**: RandomForest (200 estimators)
- **Balancing**: SMOTE + class_weight='balanced'
- **Validation**: Stratified train-test split
- **Features**: PCA-reduced feature space (95% variance)

## 🏢 Business Impact

This system enables banks to:
- Automate loan approval decisions
- Reduce manual underwriting time
- Standardize risk assessment criteria
- Improve portfolio quality through data-driven decisions
- Maintain regulatory compliance through transparent feature engineering

## 📋 Future Enhancements

- Integration with OLLAMA/LLM models for explanatory AI
- Docker containerization for scalable deployment
- OpenWebUI integration for collaborative model management
- Real-time database connectivity
- Advanced ensemble methods and deep learning models

## 🤝 Collaboration

This project was developed as part of a 3-person team during an NBK Egypt internship:
- Data extraction and preprocessing
- Feature engineering and model development
- DevOps integration with OLLAMA and Docker
- Streamlit UI development and deployment

---

**Note**: This repository contains one of multiple projects completed during the NBK Egypt internship. The data used is anonymized and processed according to banking privacy standards.
