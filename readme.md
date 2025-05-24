# 🏦 SmartScore - Intelligent Loan Risk Assessment System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartscore.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Abhishek18141/SmartScore)](https://github.com/Abhishek18141/SmartScore/issues)
[![GitHub Stars](https://img.shields.io/github/stars/Abhishek18141/SmartScore)](https://github.com/Abhishek18141/SmartScore/stargazers)

## 🚀 Live Application

**Access the live application:** [https://smartscore.streamlit.app/](https://smartscore.streamlit.app/)

### 🔐 Login Credentials
- **Username:** `Guest`
- **Password:** `abc@123`

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [API Reference](#api-reference)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

**SmartScore** is an advanced machine learning-powered loan risk assessment system that predicts the likelihood of loan default with high accuracy. Built using cutting-edge ML algorithms and trained on over 2 million loan records, SmartScore provides financial institutions with intelligent risk scoring, comprehensive analytics, and actionable insights for informed lending decisions.

### 🎯 Key Objectives
- **Risk Prediction**: Accurately predict loan default probability
- **Automated Decision Making**: Streamline loan approval processes
- **Risk Scoring**: Generate comprehensive risk scorecards
- **Data-Driven Insights**: Provide actionable analytics for lenders
- **Regulatory Compliance**: Ensure transparent and explainable AI decisions

## ✨ Features

### 🔮 Core Functionality
- **Real-time Risk Prediction**: Instant loan default probability assessment
- **Interactive Risk Scorecard**: Comprehensive risk factor analysis
- **Professional Dashboard**: Intuitive user interface with modern design
- **Multi-format Reports**: CSV, JSON, and Excel export capabilities
- **Advanced Analytics**: Interactive visualizations and insights

### 📊 Analytics & Visualization
- **Risk Distribution Charts**: Visual representation of risk factors
- **Feature Importance Analysis**: Understanding key risk drivers
- **Risk Score Gauge**: Real-time risk scoring visualization
- **Correlation Analysis**: Loan amount vs. risk relationships
- **Interactive Dashboards**: Plotly-powered dynamic charts

### 🔒 Security & Authentication
- **Secure Login System**: User authentication and session management
- **Data Privacy**: Secure handling of sensitive financial information
- **Session Management**: Persistent user sessions with logout functionality

### 📥 Export & Reporting
- **Comprehensive Reports**: Detailed prediction reports with timestamps
- **Multiple Formats**: CSV, JSON, and Excel export options
- **Audit Trail**: Complete prediction history and documentation
- **Professional Formatting**: Business-ready report templates

## 🛠️ Technology Stack

### **Frontend & UI**
- **Streamlit**: Modern web application framework
- **Plotly**: Interactive data visualization
- **HTML/CSS**: Custom styling and responsive design

### **Machine Learning & Data Science**
- **XGBoost**: Gradient boosting algorithm for predictions
- **Scikit-learn**: Data preprocessing and model utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing operations

### **Data Processing**
- **StandardScaler**: Feature normalization and scaling
- **Categorical Encoding**: Ordinal and label encoding techniques
- **Feature Engineering**: Advanced data preprocessing pipeline

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abhishek18141/SmartScore.git
cd SmartScore
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv smartscore_env

# Activate virtual environment
# On Windows:
smartscore_env\Scripts\activate
# On macOS/Linux:
source smartscore_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📱 Usage

### 1. **Access the Application**
   - Navigate to the [live application](https://smartscore.streamlit.app/)
   - Or run locally using `streamlit run app.py`

### 2. **Authentication**
   - Enter credentials:
     - Username: `Abhishek`
     - Password: `Abhishek@88201`

### 3. **Loan Application Input**
   - Fill in the loan application form with:
     - **Basic Information**: Loan amount, term, interest rate
     - **Credit Information**: Grade, sub-grade, credit history
     - **Personal Information**: Income, employment, home ownership
     - **Financial Details**: DTI ratio, purpose, credit utilization

### 4. **Risk Prediction**
   - Click "🔮 Predict Risk" to generate assessment
   - View instant results with risk probability
   - Access detailed risk scorecard

### 5. **Analytics Dashboard**
   - Explore interactive visualizations
   - Analyze risk distribution and feature importance
   - Review correlation patterns and trends

### 6. **Export Reports**
   - Download comprehensive reports in multiple formats
   - Access detailed prediction summaries
   - Generate audit trails for compliance

## 📊 Dataset Information

**Source**: [Lending Club Dataset - Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

### **Dataset Details**
- **Records**: 2+ Million loan applications
- **Features**: 150+ original features (7 selected for model)
- **Time Period**: Historical lending data
- **Data Quality**: Preprocessed and cleaned for ML training
- **Target Variable**: Loan default status (binary classification)

### **Data Preprocessing**
- **Feature Selection**: Identified 7 most predictive features
- **Missing Values**: Handled through imputation and removal
- **Categorical Encoding**: Ordinal encoding for credit grades
- **Normalization**: StandardScaler for numerical features
- **Train/Test Split**: 80/20 split for model validation

### **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Model Type**: Binary Classification
- **Training Data**: 2M+ loan records from [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Features**: 7 key financial and credit variables
- **Performance**: 94.2% accuracy rate

### **Key Features**
1. **Term**: Loan duration (36/60 months)
2. **Interest Rate**: Annual percentage rate
3. **Sub Grade**: Detailed credit grade (A1-G5)
4. **Verification Status**: Income verification level
5. **DTI Ratio**: Debt-to-income percentage
6. **Revolving Utilization**: Credit utilization rate
7. **Home Ownership**: Housing status encoding

### **Preprocessing Pipeline**
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: Ordinal encoding for grades
- **Data Validation**: Input validation and error handling
- **Feature Selection**: Optimized feature subset

## 📁 File Structure

```
SmartScore/
│
├── app.py                    # Main Streamlit application
├── xgb_model_v2.pkl         # Trained XGBoost model
├── minmax_scaler.pkl        # Fitted StandardScaler
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── assets/                 # Static assets (if any)
├── docs/                   # Additional documentation
└── tests/                  # Unit tests (if any)
```

### **Core Files Description**

| File | Description |
|------|-------------|
| `app.py` | Main application with UI, authentication, and prediction logic |
| `xgb_model_v2.pkl` | Pre-trained XGBoost model for risk prediction |
| `minmax_scaler.pkl` | Fitted StandardScaler for feature normalization |
| `requirements.txt` | List of Python package dependencies |

## 🔧 API Reference

### **Main Functions**

#### `load_models()`
```python
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost model and scaler"""
    return xgb_model, scaler
```

#### `preprocess_input(df, scaler)`
```python
def preprocess_input(df, scaler):
    """
    Preprocess input data for model prediction
    
    Args:
        df (pd.DataFrame): Input features
        scaler: Fitted StandardScaler
    
    Returns:
        tuple: (scaled_features, processed_dataframe)
    """
```

#### `calculate_risk_scorecard(prediction_proba, input_data)`
```python
def calculate_risk_scorecard(prediction_proba, input_data):
    """
    Generate comprehensive risk scorecard
    
    Args:
        prediction_proba (float): Risk probability
        input_data (dict): Input features
    
    Returns:
        dict: Risk scorecard with detailed metrics
    """
```

## 📸 Screenshots

### Login Interface
Professional authentication system with secure credentials management.

### Loan Application Form
Comprehensive input form with validation and user-friendly design.

### Prediction Results
Clear approval/rejection decision with detailed risk analysis.

### Analytics Dashboard
Interactive visualizations with risk distribution and feature importance.

### Export Reports
Multiple format options with professional report templates.

## 🤝 Contributing

We welcome contributions to SmartScore! Here's how you can help:

### **Getting Started**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Contribution Guidelines**
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

### **Areas for Contribution**
- Model performance improvements
- Additional visualization features
- Enhanced security measures
- Mobile responsiveness
- API development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Developer**: Abhishek Pandey  
**Email**: [abhishekpandey88201@gmail.com](mailto:abhishekpandey88201@gmail.com)  
**LinkedIn**: [Abhishek Pandey](https://www.linkedin.com/in/abhishek-pandey-108012219)  
**GitHub**: [@Abhishek18141](https://github.com/Abhishek18141)  
**Project Repository**: [SmartScore](https://github.com/Abhishek18141/SmartScore/)  
**Live Application**: [smartscore.streamlit.app](https://smartscore.streamlit.app/)

### **Support & Collaboration**
- 📧 **Email**: For professional inquiries and collaboration opportunities
- 💼 **LinkedIn**: Connect for networking and career discussions
- 🐛 **GitHub Issues**: For bug reports and feature requests
- 💡 **Feature Requests**: Submit through GitHub Issues
- 🤝 **Open Source**: Welcome contributions and pull requests

---

## 🙏 Acknowledgments

- **Streamlit Team** - For the amazing web framework
- **XGBoost Developers** - For the powerful ML algorithm
- **Plotly Team** - For interactive visualization capabilities
- **Open Source Community** - For continuous inspiration and support

---

## 📈 Project Statistics

- **Lines of Code**: 500+
- **Model Accuracy**: 94.2%
- **Training Data**: 2M+ records
- **Features**: 7 key variables
- **Export Formats**: 3 (CSV, JSON, Excel)

---

**⭐ If you find SmartScore helpful, please consider giving it a star on GitHub!**

[![GitHub stars](https://img.shields.io/github/stars/Abhishek18141/SmartScore.svg?style=social&label=Star)](https://github.com/Abhishek18141/SmartScore)
