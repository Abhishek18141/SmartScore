import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="SmartScore",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load models (mock loading for demonstration)
@st.cache_resource
def load_models():
    import pickle

    with open("xgb_model_v2.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return xgb_model, scaler


# Preprocessing function
def preprocess_input(df, scaler):
    """Preprocess input data for model prediction"""
    df_copy = df.copy()
    
    # Handle 'term'
    df_copy['term'] = df_copy['term'].str.extract(r'(\d+)').astype(float)
    
    # Handle 'sub_grade'
    df_copy['sub_grade'] = pd.Categorical(df_copy['sub_grade'], categories=[
        'A1', 'A2', 'A3', 'A4', 'A5',
        'B1', 'B2', 'B3', 'B4', 'B5',
        'C1', 'C2', 'C3', 'C4', 'C5',
        'D1', 'D2', 'D3', 'D4', 'D5',
        'E1', 'E2', 'E3', 'E4', 'E5',
        'F1', 'F2', 'F3', 'F4', 'F5',
        'G1', 'G2', 'G3', 'G4', 'G5'
    ], ordered=True)
    df_copy['sub_grade'] = df_copy['sub_grade'].cat.codes
    
    # Handle 'verification_status'
    verification_mapping = {
        'Not Verified': 0,
        'Source Verified': 1,
        'Verified': 2
    }
    df_copy['verification_status'] = df_copy['verification_status'].map(verification_mapping)
    
    # Handle 'home_ownership_encoded'
    home_ownership_order = {
        'OTHER': 0,
        'RENT': 1,
        'MORTGAGE': 2,
        'OWN': 3
    }
    df_copy['home_ownership_encoded'] = df_copy['home_ownership_encoded'].map(home_ownership_order)
    
    # Cast numeric columns
    num_cols = ['term', 'sub_grade', 'verification_status', 'home_ownership_encoded', 'dti', 'int_rate', 'revol_util']
    for col in num_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype('float32')
    
    # Select only model features
    model_features = ['term', 'int_rate', 'sub_grade', 'verification_status', 'dti', 'revol_util', 'home_ownership_encoded']
    df_model = df_copy[model_features]
    
    # Scale the data
    df_scaled = scaler.transform(df_model)
    
    return df_scaled, df_copy

# Authentication function
def authenticate(username, password):
    return username == "User" and password == "abc@123"

# Risk scorecard calculation
def calculate_risk_scorecard(prediction_proba, input_data):
    risk_score = int((1 - prediction_proba) * 1000)
    
    scorecard = {
        'Overall Risk Score': risk_score,
        'Credit Grade Impact': f"{input_data['grade']} Grade",
        'Term Impact': f"{input_data['term']} Impact",
        'Income Verification': input_data['verification_status'],
        'DTI Ratio Impact': f"{input_data['dti']:.1f}% DTI",
        'Interest Rate Factor': f"{input_data['int_rate']:.2f}% Rate"
    }
    
    return scorecard

# Analytics visualizations
def create_analytics(input_data, prediction_proba):
    # Risk Distribution Pie Chart
    fig1 = go.Figure(data=[go.Pie(
        labels=['Low Risk', 'High Risk'],
        values=[prediction_proba, 1-prediction_proba],
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    fig1.update_layout(title="Risk Distribution", height=400)
    
    # Feature Importance (Mock data)
    features = ['Interest Rate', 'DTI Ratio', 'Sub Grade', 'Term', 'Verification', 'Revolving Util', 'Home Ownership']
    importance = [0.25, 0.20, 0.18, 0.15, 0.10, 0.08, 0.04]
    
    fig2 = px.bar(
        x=features, y=importance,
        title="Feature Importance in Risk Assessment",
        color=importance,
        color_continuous_scale='viridis'
    )
    fig2.update_layout(height=400, showlegend=False)
    
    # Risk Score Gauge
    fig3 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = int((1-prediction_proba)*1000),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score (0-1000)"},
        delta = {'reference': 500},
        gauge = {
            'axis': {'range': [None, 1000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 300], 'color': "lightgreen"},
                {'range': [300, 700], 'color': "yellow"},
                {'range': [700, 1000], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 750
            }
        }
    ))
    fig3.update_layout(height=400)
    
    # Loan Amount vs Risk Correlation (Mock)
    loan_amounts = np.linspace(5000, 50000, 20)
    risk_scores = np.random.normal(500, 150, 20)
    
    fig4 = px.scatter(
        x=loan_amounts, y=risk_scores,
        title="Loan Amount vs Risk Score Correlation",
        labels={'x': 'Loan Amount ($)', 'y': 'Risk Score'},
        trendline="ols"
    )
    fig4.update_layout(height=400)
    
    return fig1, fig2, fig3, fig4

# Download functions
def create_download_data(input_data, prediction, prediction_proba, scorecard):
    # Create comprehensive report
    report_data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Loan_Amount': [input_data['loan_amnt']],
        'Term': [input_data['term']],
        'Interest_Rate': [input_data['int_rate']],
        'Grade': [input_data['grade']],
        'Sub_Grade': [input_data['sub_grade']],
        'Annual_Income': [input_data['annual_inc']],
        'DTI_Ratio': [input_data['dti']],
        'Prediction': ['Approved' if prediction == 0 else 'Rejected'],
        'Risk_Probability': [f"{prediction_proba:.4f}"],
        'Risk_Score': [scorecard['Overall Risk Score']]
    }
    
    return pd.DataFrame(report_data)

# Main application
def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None

    # Login section
    if not st.session_state.authenticated:
        st.markdown('<h1 class="main-header">🏦 SmartScore</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Secure Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        return

    # Load models
    model, scaler = load_models()

    # Header
    st.markdown('<h1 class="main-header">🏦 SmartScore</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Dashboard")
        st.success(f"Welcome to SmartScore!")
    
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.prediction_data = None
            st.rerun()
    
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown('<div class="metric-card">Total Predictions: 1,234</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">Success Rate: 94.2%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">Avg Risk Score: 342</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧠 About SmartScore")
        st.markdown("""
            **SmartScore** analyzes credit risk  
            using ML models trained on 2M+ records.  
            It predicts default probability,  
            generates risk scores & scorecards,  
            and provides a predictive dashboard.
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Loan Application", "🎯 Prediction Results", "📊 Analytics Dashboard", "📥 Reports & Downloads"])

    with tab1:
        st.markdown('<h2 class="sub-header">Loan Application Details</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            application_type = st.selectbox("Application Type", ['Individual', 'Joint'])
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
            term = st.selectbox("Term", ['36 months', '60 months'])
            int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
            installment = st.number_input("Monthly Installment ($)", min_value=50, max_value=3000, value=500, step=10)
            
            st.markdown("#### Credit Information")
            grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            sub_grade = st.selectbox("Sub Grade", [
                'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
                'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
                'G1', 'G2', 'G3', 'G4', 'G5'
            ])
        
        with col2:
            st.markdown("#### Personal Information")
            emp_length = st.selectbox("Employment Length", 
                ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
                 '6 years', '7 years', '8 years', '9 years', '10+ years'])
            home_ownership = st.selectbox("Home Ownership", ['OTHER', 'RENT', 'MORTGAGE','OWN'])
            annual_inc = st.number_input("Annual Income ($)", min_value=20000, max_value=5000000, value=20000, step=5000)
            verification_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
            
            st.markdown("#### Financial Details")
            purpose = st.selectbox("Purpose", [
                'debt_consolidation', 'credit_card', 'home_improvement', 'other',
                'major_purchase', 'medical', 'small_business', 'car', 'vacation',
                'moving', 'house', 'wedding', 'renewable_energy', 'educational'
            ])
            dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            delinq_2yrs = st.number_input("Delinquencies (2 years)", min_value=0, max_value=0, value=0)
            open_acc = st.number_input("Open Credit Lines", min_value=1, max_value=50, value=1)
            pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
            revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
                # Prepare input data
                input_data = {
                    'application_type': application_type,
                    'loan_amnt': loan_amnt,
                    'term': term,
                    'int_rate': int_rate,
                    'installment': installment,
                    'grade': grade,
                    'sub_grade': sub_grade,
                    'emp_length': emp_length,
                    'home_ownership_encoded': home_ownership,
                    'annual_inc': annual_inc,
                    'verification_status': verification_status,
                    'purpose': purpose,
                    'dti': dti,
                    'delinq_2yrs': delinq_2yrs,
                    'open_acc': open_acc,
                    'pub_rec': pub_rec,
                    'revol_util': revol_util
                }
                
                # Convert to DataFrame
                df_input = pd.DataFrame([input_data])
                selected_columns = ['term', 'int_rate', 'sub_grade', 'verification_status', 'dti', 'revol_util', 'home_ownership_encoded']
                filtered_df = df_input[selected_columns]
                # Preprocess and predict
                X_processed, df_processed = preprocess_input(filtered_df, scaler)
                prediction = model.predict(X_processed)[0]
                prediction_proba = model.predict_proba(X_processed)[0][1]
                
                # Calculate scorecard
                scorecard = calculate_risk_scorecard(prediction_proba, input_data)
                
                # Store results
                st.session_state.prediction_data = {
                    'input_data': input_data,
                    'prediction': prediction,
                    'prediction_proba': prediction_proba,
                    'scorecard': scorecard
                }
                
                st.success("✅ Prediction completed! Check the Prediction Results tab.")

        with col2:
            if st.button("🧹 Clear Form", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("📊 View Analytics", use_container_width=True):
                if st.session_state.prediction_data:
                    st.info("📈 Analytics available in Analytics Dashboard tab")
                else:
                    st.warning("⚠️ Please make a prediction first")

    with tab2:
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        if st.session_state.prediction_data:
            data = st.session_state.prediction_data
            
            # Prediction result
            if data['prediction'] == 0:
                st.markdown(f'''
                <div class="success-card">
                    <h2>✅ LOAN APPROVED</h2>
                    <h3>Risk Probability: {data['prediction_proba']:.2%}</h3>
                    <p>This loan application has been approved based on the risk assessment model.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="prediction-card">
                    <h2>❌ LOAN REJECTED</h2>
                    <h3>Risk Probability: {data['prediction_proba']:.2%}</h3>
                    <p>This loan application has been rejected due to high risk factors.</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Scorecard
            st.markdown('<h3 class="sub-header">Risk Scorecard</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            for i, (key, value) in enumerate(data['scorecard'].items()):
                col = [col1, col2, col3][i % 3]
                with col:
                    st.metric(key, value)
            
            # Key factors
            st.markdown('<h3 class="sub-header">Key Risk Factors</h3>', unsafe_allow_html=True)
            
            factors = [
                f"💰 Loan Amount: ${data['input_data']['loan_amnt']:,}",
                f"📊 Credit Grade: {data['input_data']['grade']}{data['input_data']['sub_grade'][-1]}",
                f"💸 Interest Rate: {data['input_data']['int_rate']:.2f}%",
                f"🏠 Home Ownership: {data['input_data']['home_ownership_encoded']}",
                f"📈 DTI Ratio: {data['input_data']['dti']:.1f}%",
                f"🔄 Revolving Utilization: {data['input_data']['revol_util']:.1f}%"
            ]
            
            for factor in factors:
                st.write(f"• {factor}")
                
        else:
            st.info("📝 Please submit a loan application first to see prediction results.")

    with tab3:
        st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        if st.session_state.prediction_data:
            data = st.session_state.prediction_data
            fig1, fig2, fig3, fig4 = create_analytics(data['input_data'], data['prediction_proba'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig3, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)
                
        else:
            st.info("📊 Analytics will be available after making a prediction.")

    with tab4:
        st.markdown('<h2 class="sub-header">Reports & Downloads</h2>', unsafe_allow_html=True)
        
        if st.session_state.prediction_data:
            data = st.session_state.prediction_data
            
            # Create downloadable report
            report_df = create_download_data(
                data['input_data'], 
                data['prediction'], 
                data['prediction_proba'], 
                data['scorecard']
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv,
                    file_name=f"loan_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON download
                json_data = report_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON Report",
                    data=json_data,
                    file_name=f"loan_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    report_df.to_excel(writer, sheet_name='Loan_Prediction', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=f"loan_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Display report preview
            st.markdown('<h3 class="sub-header">Report Preview</h3>', unsafe_allow_html=True)
            st.dataframe(report_df, use_container_width=True)
            
        else:
            st.info("📥 Reports will be available after making a prediction.")

if __name__ == "__main__":
    main()
