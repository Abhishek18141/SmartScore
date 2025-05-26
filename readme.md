# 🏦 SmartScore - Intelligent Loan Risk Assessment System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartscore.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Abhishek18141/SmartScore)](https://github.com/Abhishek18141/SmartScore/issues)
[![GitHub Stars](https://img.shields.io/github/stars/Abhishek18141/SmartScore)](https://github.com/Abhishek18141/SmartScore/stargazers)
[![Machine Learning](https://img.shields.io/badge/ML-Credit%20Risk-green.svg)](https://github.com/Abhishek18141/SmartScore/)

## 🚀 Live Application
**Access the application:** [https://smartscore.streamlit.app/](https://smartscore.streamlit.app/)

### Demo Credentials
- **Username:** `User`
- **Password:** `Abc@123`

## 📋 Overview

SmartScore is an intelligent credit risk assessment system that leverages machine learning to evaluate loan approval probability. The application provides real-time credit scoring capabilities through an intuitive web interface, helping financial institutions make data-driven lending decisions.

## ✨ Features

- **Dual Prediction Modes**: Single application assessment and bulk processing capabilities
- **Deep Learning Model**: Multilayer Perceptron (MLP) with 97.65% accuracy trained on 4,269+ records
- **Real-time Analytics**: Interactive dashboards with Plotly visualizations
- **Comprehensive Scoring**: Credit scorecard (0-1000) and probability assessment
- **Advanced Visualizations**: 
  - Gauge charts for scorecard display
  - Feature importance analysis
  - Asset distribution breakdowns
  - Approval rate analytics
- **Bulk Processing**: Upload CSV files for multiple loan applications
- **Export Capabilities**: Download results in CSV/Excel formats with embedded charts
- **Secure Authentication**: Protected access with user credentials
- **Professional UI**: Modern styling with gradient designs and responsive layout

## 🏗️ Project Structure

```
SmartScore/
├── app.py                           # Main Streamlit application
├── credit_risk_model.h5            # Trained machine learning model
├── credit_risk_model_building.ipynb # Model development notebook
├── loan_approval_dataset.csv       # Training dataset
├── requirements.txt                # Python dependencies
├── scaler.pkl                      # Feature scaling transformer
└── README.md                       # Project documentation
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Visualization**: Plotly Express & Plotly Graph Objects
- **Machine Learning**: TensorFlow/Keras (Deep Learning MLP)
- **Data Processing**: Pandas, NumPy
- **Model Deployment**: Scikit-learn (preprocessing), Joblib (model serialization)
- **File Processing**: XlsxWriter for Excel export
- **Authentication**: Session-based login system
- **Deployment**: Streamlit Cloud

## 📊 Dataset

The model is trained on a comprehensive loan approval dataset containing various financial and demographic features.

**Dataset Source**: [Kaggle - Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

### Key Features:
- **Comprehensive Analysis**: 11 input variables covering demographics, financials, and assets
- **Advanced Preprocessing**: Feature scaling and categorical encoding
- **Risk Categorization**: Multi-tier scoring system (0-400: High Risk, 400-600: Medium Risk, 600-800: Low Risk, 800-1000: Excellent)
- **Real-time Processing**: Instant predictions for both single and bulk applications

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abhishek18141/SmartScore.git
   cd SmartScore
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Use the demo credentials provided above

## 🔧 Usage

### Single Loan Prediction
1. **Login**: Use credentials (Username: `User`, Password: `Abc@123`)
2. **Fill Application**: Complete the loan application form with:
   - Personal Information (dependents, education, employment)
   - Financial Details (income, loan amount, term, CIBIL score)
   - Asset Information (residential, commercial, luxury, bank assets)
3. **Get Results**: Receive instant approval/rejection with:
   - Credit scorecard (0-1000 scale)
   - Approval probability percentage
   - Interactive dashboard with feature importance
   - Asset distribution analysis
4. **Download Reports**: Export application data, dashboard charts, or prediction reports

### Bulk Loan Processing
1. **Download Template**: Get the demo CSV format with required columns
2. **Prepare Data**: Fill your loan applications in the template format
3. **Upload & Process**: Upload CSV file for batch processing
4. **Analyze Results**: View comprehensive analytics dashboard including:
   - Overall approval rates and statistics
   - Scorecard distribution histograms
   - CIBIL score vs approval probability scatter plots
   - Education and employment analysis
5. **Export Reports**: Download complete results, approved/rejected lists, or summary reports

## 📈 Model Performance

The deep learning credit risk model demonstrates exceptional performance:
- **Architecture**: Multilayer Perceptron (MLP) neural network
- **Accuracy**: 97.65% on validation data
- **Training Data**: 4,269+ loan application records
- **Features**: 11 input variables including financial and demographic data
- **Scoring Range**: 0-1000 credit scorecard with probability-based assessment
- **Feature Importance**: CIBIL Score (77%), Loan Term (11%), Assets (5%), Income (3%), Education (2%), Loan Amount (2%)

*Detailed model architecture and validation metrics are available in the Jupyter notebook.*

## 🔐 Security Features

- Secure user authentication
- Data privacy protection
- Input validation and sanitization
- Session management

## 🚀 Deployment

The application is deployed on Streamlit Cloud for easy access and scalability. The deployment includes:
- Automatic updates from the main branch
- Environment configuration
- Dependency management
- Performance monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contact & Support

**Developer**: Abhishek Pandey

- **Email**: [abhishekpandey88201@gmail.com](mailto:abhishekpandey88201@gmail.com)
- **LinkedIn**: [Abhishek Pandey](https://www.linkedin.com/in/abhishek-pandey-108012219)
- **GitHub**: [Abhishek18141](https://github.com/Abhishek18141)

## 🙏 Acknowledgments

- Dataset provided by [Archit Sharma](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) on Kaggle
- Streamlit community for the excellent deployment platform
- Open source libraries that made this project possible

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/Abhishek18141/SmartScore?style=social)
![GitHub forks](https://img.shields.io/github/forks/Abhishek18141/SmartScore?style=social)
![GitHub issues](https://img.shields.io/github/issues/Abhishek18141/SmartScore)

---

**⭐ If you find SmartScore helpful, please consider giving it a star on GitHub!**

[![GitHub stars](https://img.shields.io/github/stars/Abhishek18141/SmartScore.svg?style=social&label=Star)](https://github.com/Abhishek18141/SmartScore)

---
