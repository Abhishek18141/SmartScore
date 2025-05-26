import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
from datetime import datetime
import base64
from tensorflow.keras.models import load_model
import joblib
from io import BytesIO
import xlsxwriter
import tempfile

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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .rejected {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .bulk-summary {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'bulk_prediction_made' not in st.session_state:
    st.session_state.bulk_prediction_made = False
if 'bulk_prediction_data' not in st.session_state:
    st.session_state.bulk_prediction_data = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Single Prediction"

def login_page():
    """Display login page."""
    st.markdown('<div class="main-header">🏦 SmartScore</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        if st.button("Login", key="login_btn"):
            if username == "User" and password == "Abc@123":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials! Please use Username: 'User' and Password: 'Abc@123'")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # credentials display
        st.info("**Credentials:**\n\nUsername: User\n\nPassword: Abc@123")

def create_demo_format():
    """Create demo format for bulk upload."""
    demo_data = {
        'applicant_id': ['APP001', 'APP002', 'APP003'],
        'no_of_dependents': [2, 0, 1],
        'education': ['Graduate', 'Not Graduate', 'Graduate'],
        'self_employed': ['No', 'Yes', 'No'],
        'income_annum': [500000, 300000, 800000],
        'loan_amount': [2000000, 1500000, 3000000],
        'loan_term': [15, 10, 20],
        'cibil_score': [750, 650, 800],
        'residential_assets_value': [2000000, 1000000, 3000000],
        'commercial_assets_value': [500000, 0, 1000000],
        'luxury_assets_value': [300000, 100000, 500000],
        'bank_asset_value': [200000, 50000, 400000]
    }
    return pd.DataFrame(demo_data)

def process_bulk_predictions(df):
    """Process bulk predictions."""
    try:
        # Load model and scaler
        model = load_model("credit_risk_model.h5")
        scaler = joblib.load('scaler.pkl')
        
        results = []
        
        for index, row in df.iterrows():
            features = {
                'no_of_dependents': row['no_of_dependents'],
                'education': row['education'],
                'self_employed': row['self_employed'],
                'income_annum': row['income_annum'],
                'loan_amount': row['loan_amount'],
                'loan_term': row['loan_term'],
                'cibil_score': row['cibil_score'],
                'residential_assets_value': row['residential_assets_value'],
                'commercial_assets_value': row['commercial_assets_value'],
                'luxury_assets_value': row['luxury_assets_value'],
                'bank_asset_value': row['bank_asset_value']
            }
            
            # Preprocessing
            df_single = pd.DataFrame([features])
            df_single['education'] = df_single['education'].map({'Graduate': 1, 'Not Graduate': 0})
            df_single['self_employed'] = df_single['self_employed'].map({'Yes': 1, 'No': 0})
            input_data = scaler.transform(df_single)
            prediction = model.predict(input_data)
            
            # Calculate results
            approval = True if prediction >= 0.5 else False
            scorecard = int(prediction * 1000)
            probability = float(prediction)
            
            result = {
                'applicant_id': row['applicant_id'] if 'applicant_id' in row else f"APP{index+1:03d}",
                'approval': approval,
                'approval_status': 'Approved' if approval else 'Rejected',
                'scorecard': scorecard,
                'probability': probability,
                'probability_percent': f"{probability:.2%}",
                **features
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error processing predictions: {str(e)}")
        return None

def bulk_prediction_page():
    """Display bulk prediction page."""
    st.markdown('<div class="main-header">📊 Bulk Loan Prediction</div>', unsafe_allow_html=True)
    
    # Step 1: Download Demo Format
    st.markdown("## 📥 Step 1: Download Demo Format")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Download the demo format, fill in your data, and upload it back for bulk predictions.")
    
    with col2:
        demo_df = create_demo_format()
        csv_demo = demo_df.to_csv(index=False)
        st.download_button(
            label="📋 Download Demo Format (CSV)",
            data=csv_demo,
            file_name=f"bulk_prediction_demo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Display demo format
    with st.expander("👁️ Preview Demo Format"):
        st.dataframe(demo_df, use_container_width=True)
    
    st.markdown("---")
    
    # Step 2: Upload Data
    st.markdown("## 📤 Step 2: Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with loan applications",
        type=['csv'],
        help="Upload a CSV file with the same format as the demo file"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df_upload = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = [
                'no_of_dependents', 'education', 'self_employed', 'income_annum',
                'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
            ]
            
            missing_columns = [col for col in required_columns if col not in df_upload.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your file has all required columns as shown in the demo format.")
            else:
                st.success(f"✅ File uploaded successfully! Found {len(df_upload)} applications.")
                
                # Preview uploaded data
                with st.expander("👁️ Preview Uploaded Data"):
                    st.dataframe(df_upload.head(10), use_container_width=True)
                
                # Process predictions
                if st.button("🔮 Process Bulk Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions... This may take a few moments."):
                        results_df = process_bulk_predictions(df_upload)
                        
                        if results_df is not None:
                            st.session_state.bulk_prediction_made = True
                            st.session_state.bulk_prediction_data = results_df
                            st.success("✅ Bulk predictions completed successfully!")
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV with the correct format.")

def bulk_results_dashboard():
    """Display bulk prediction results dashboard."""
    df = st.session_state.bulk_prediction_data
    
    st.markdown('<div class="main-header">📊 Bulk Prediction Results</div>', unsafe_allow_html=True)
    
    # Summary Statistics
    total_applications = len(df)
    approved_count = len(df[df['approval'] == True])
    rejected_count = total_applications - approved_count
    approval_rate = approved_count / total_applications
    avg_scorecard = df['scorecard'].mean()
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", total_applications)
    with col2:
        st.metric("Approved", approved_count, f"{approval_rate:.1%}")
    with col3:
        st.metric("Rejected", rejected_count, f"{1-approval_rate:.1%}")
    with col4:
        st.metric("Avg Scorecard", f"{avg_scorecard:.0f}")
    
    st.markdown(f'<div class="bulk-summary">Approval Rate: {approval_rate:.1%} | Average Scorecard: {avg_scorecard:.0f}/1000</div>', unsafe_allow_html=True)
    
    # Dashboard Charts
    st.markdown("## 📈 Analytics Dashboard")
    
    # Row 1: Approval Distribution and Scorecard Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval Pie Chart
        approval_counts = df['approval_status'].value_counts()
        fig_pie = px.pie(
            values=approval_counts.values,
            names=approval_counts.index,
            title="Loan Approval Distribution",
            color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#ff6b6b'}
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Scorecard Distribution Histogram
        fig_hist = px.histogram(
            df, x='scorecard', nbins=20,
            title="Scorecard Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 2: CIBIL Score vs Approval and Income vs Loan Amount
    col1, col2 = st.columns(2)
    
    with col1:
        # CIBIL Score vs Approval
        fig_scatter1 = px.scatter(
            df, x='cibil_score', y='probability',
            color='approval_status',
            title="CIBIL Score vs Approval Probability",
            color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#ff6b6b'}
        )
        fig_scatter1.update_layout(height=400)
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        # Income vs Loan Amount
        fig_scatter2 = px.scatter(
            df, x='income_annum', y='loan_amount',
            color='approval_status', size='scorecard',
            title="Income vs Loan Amount (Size = Scorecard)",
            color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#ff6b6b'}
        )
        fig_scatter2.update_layout(height=400)
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Row 3: Feature Analysis
    st.markdown("### 🔍 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Education vs Approval
        edu_approval = df.groupby(['education', 'approval_status']).size().unstack(fill_value=0)
        fig_bar1 = px.bar(
            edu_approval, 
            title="Approval by Education Level",
            color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#ff6b6b'}
        )
        fig_bar1.update_layout(height=300)
        st.plotly_chart(fig_bar1, use_container_width=True)
    
    with col2:
        # Self Employment vs Approval
        emp_approval = df.groupby(['self_employed', 'approval_status']).size().unstack(fill_value=0)
        fig_bar2 = px.bar(
            emp_approval,
            title="Approval by Employment Type",
            color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#ff6b6b'}
        )
        fig_bar2.update_layout(height=300)
        st.plotly_chart(fig_bar2, use_container_width=True)
    
    # Detailed Results Table
    st.markdown("## 📋 Detailed Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "Approved", "Rejected"])
    with col2:
        min_scorecard = st.slider("Minimum Scorecard", 0, 1000, 0)
    with col3:
        show_rows = st.selectbox("Rows to display", [10, 25, 50, 100, "All"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['approval_status'] == status_filter]
    
    filtered_df = filtered_df[filtered_df['scorecard'] >= min_scorecard]
    
    if show_rows != "All":
        filtered_df = filtered_df.head(show_rows)
    
    # Display table
    display_columns = [
        'applicant_id', 'approval_status', 'scorecard', 'probability_percent',
        'cibil_score', 'income_annum', 'loan_amount', 'loan_term'
    ]
    
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Download Options
    st.markdown("## 📥 Download Reports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download Complete Results
        csv_complete = df.to_csv(index=False)
        st.download_button(
            label="📊 Complete Results (CSV)",
            data=csv_complete,
            file_name=f"bulk_predictions_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download Approved Only
        approved_df = df[df['approval'] == True]
        csv_approved = approved_df.to_csv(index=False)
        st.download_button(
            label="✅ Approved Only (CSV)",
            data=csv_approved,
            file_name=f"bulk_predictions_approved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Download Rejected Only
        rejected_df = df[df['approval'] == False]
        csv_rejected = rejected_df.to_csv(index=False)
        st.download_button(
            label="❌ Rejected Only (CSV)",
            data=csv_rejected,
            file_name=f"bulk_predictions_rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col4:
        # Download Summary Report
        summary_data = {
            'Metric': ['Total Applications', 'Approved', 'Rejected', 'Approval Rate', 'Average Scorecard'],
            'Value': [total_applications, approved_count, rejected_count, f"{approval_rate:.2%}", f"{avg_scorecard:.0f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📈 Summary Report (CSV)",
            data=csv_summary,
            file_name=f"bulk_predictions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Reset button
    if st.button("🔄 New Bulk Prediction", type="primary"):
        st.session_state.bulk_prediction_made = False
        st.session_state.bulk_prediction_data = None
        st.rerun()

def loan_application_form():
    """Display loan application form."""
    st.markdown('<div class="main-header">📋 Loan Application Form</div>', unsafe_allow_html=True)
    
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            income_annum = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=0)
        
        with col2:
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount (₹)", min_value=50000, max_value=40000000, value=50000)
            loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=20, value=1)
            cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900, value=0)
        
        st.subheader("Asset Information")
        col3, col4 = st.columns(2)
        
        with col3:
            residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=0)
            commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0)
        
        with col4:
            luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, max_value=40000000, value=0)
            bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=0)
        
        submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

        if submitted:
            features = {
                'no_of_dependents': no_of_dependents,
                'education': education,
                'self_employed': self_employed,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'residential_assets_value': residential_assets_value,
                'commercial_assets_value': commercial_assets_value,
                'luxury_assets_value': luxury_assets_value,
                'bank_asset_value': bank_asset_value
            }

            #Model and Scaler Loading
            model = load_model("credit_risk_model.h5")
            scaler = joblib.load('scaler.pkl')
            
            #preprocessing
            df = pd.DataFrame([features])
            df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
            df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
            input = scaler.transform(df)
            out = model.predict(input)

            # Make prediction
            approval = True if out >= 0.5 else False
            scorecard = int(out * 1000)
            probability = (float(out))

            # Store results
            st.session_state.prediction_made = True
            st.session_state.prediction_data = {
                'features': features,
                'approval': approval,
                'scorecard': scorecard,
                'probability': probability,
                'raw_inputs': {
                    'no_of_dependents': no_of_dependents,
                    'education': education,
                    'self_employed': self_employed,
                    'income_annum': income_annum,
                    'loan_amount': loan_amount,
                    'loan_term': loan_term,
                    'cibil_score': cibil_score,
                    'residential_assets_value': residential_assets_value,
                    'commercial_assets_value': commercial_assets_value,
                    'luxury_assets_value': luxury_assets_value,
                    'bank_asset_value': bank_asset_value
                }
            }
            
            st.rerun()

def display_prediction_results():
    """Display prediction results and dashboard."""
    data = st.session_state.prediction_data
    
    st.markdown('<div class="main-header">🎯 Loan Approval Results</div>', unsafe_allow_html=True)
    
    # Prediction Result
    if data['approval']:
        st.markdown(f'<div class="approved">✅ LOAN APPROVED<br>Scorecard: {data["scorecard"]}/1000</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="rejected">❌ LOAN REJECTED<br>Scorecard: {data["scorecard"]}/1000</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Approval Probability", f"{data['probability']:.2%}")
    with col2:
        st.metric("Scorecard", f"{data['scorecard']}/1000")
    with col3:
        st.metric("CIBIL Score", data['features']['cibil_score'])
    with col4:
        income_loan_ratio = data['features']['income_annum'] / data['features']['loan_amount']
        st.metric("Income/Loan Ratio", f"{income_loan_ratio:.2f}")
    
    # Dashboard
    st.markdown("## 📊 Dashboard")
    
    # Risk Assessment Chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Scorecard Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = data['scorecard'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Credit Scorecard"},
            delta = {'reference': 600},
            gauge = {
                'axis': {'range': [None, 1000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 400], 'color': "lightgray"},
                    {'range': [400, 600], 'color': "yellow"},
                    {'range': [600, 800], 'color': "lightgreen"},
                    {'range': [800, 1000], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 600}}))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Feature Importance
        feature_importance = {
            'CIBIL Score': 0.77,
            'Loan Term': 0.11,
            'Loan Amount': 0.02,
            'Assets Value': 0.05,
            'Income': 0.03,
            'Education': 0.02
        }
        
        fig_bar = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Feature Importance in Decision",
            color=list(feature_importance.values()),
            color_continuous_scale="viridis"
        )
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Asset Distribution
    assets = {
        'Residential': data['features']['residential_assets_value'],
        'Commercial': data['features']['commercial_assets_value'],
        'Luxury': data['features']['luxury_assets_value'],
        'Bank': data['features']['bank_asset_value']
    }
    
    fig_pie = px.pie(
        values=list(assets.values()),
        names=list(assets.keys()),
        title="Asset Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Downloadable Reports
    st.markdown("## 📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)

    def fig_to_image_bytes(fig):
        buf = BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return buf
    
    with col1:
        if st.button("📋 Download Application Form"):
            df_form = pd.DataFrame([data['raw_inputs']])
            csv_form = df_form.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_form,
                file_name=f"loan_application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Dashboard Data"):
            # Generate images from charts
            img_gauge = fig_to_image_bytes(fig_gauge)
            img_bar = fig_to_image_bytes(fig_bar)
            img_pie = fig_to_image_bytes(fig_pie)

            # Raw dashboard data for sheet 2
            dashboard_data = {
                'approval': data['approval'],
                'scorecard': data['scorecard'],
                'probability': data['probability'],
                **data['features']
            }
            df_dashboard = pd.DataFrame([dashboard_data])

            # Prepare Excel in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Sheet 1: Dashboard Charts
                sheet1 = workbook.add_worksheet("Dashboard Charts")
                writer.sheets["Dashboard Charts"] = sheet1

                # Insert images
                sheet1.insert_image("B2", "gauge.png", {'image_data': img_gauge, 'x_scale': 1, 'y_scale': 1})
                sheet1.insert_image("B20", "bar.png", {'image_data': img_bar, 'x_scale': 1, 'y_scale': 1})
                sheet1.insert_image("B38", "pie.png", {'image_data': img_pie, 'x_scale': 1, 'y_scale': 1})

                # Sheet 2: Raw Dashboard Data
                df_dashboard.to_excel(writer, sheet_name="Raw Data", index=False)

            output.seek(0)

            # Download button
            st.download_button(
                label="📊 Download Excel with Charts",
                data=output,
                file_name=f"dashboard_with_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("🎯 Download Prediction Report"):
            prediction_report = {
                'Application_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Loan_Status': 'Approved' if data['approval'] else 'Rejected',
                'Scorecard': data['scorecard'],
                'Probability': f"{data['probability']:.2%}",
                **data['raw_inputs']
            }
            df_prediction = pd.DataFrame([prediction_report])
            csv_prediction = df_prediction.to_csv(index=False)
            st.download_button(
                label="🎯 Download CSV",
                data=csv_prediction,
                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Reset button
    if st.button("🔄 New Application", type="primary"):
        st.session_state.prediction_made = False
        st.session_state.prediction_data = None
        st.rerun()


def main():
    """Main application logic."""
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏦 SmartScore")
        st.markdown("---")
        
        if st.session_state.logged_in:
            st.success("✅ Logged in as: User")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.prediction_made = False
                st.session_state.prediction_data = None
                st.session_state.bulk_prediction_made = False
                st.session_state.bulk_prediction_data = None
                st.rerun()
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 📋 Navigation")
            tab_option = st.radio(
                "Choose Prediction Type:",
                ["Single Prediction", "Bulk Prediction"],
                key="nav_radio"
            )
            st.session_state.current_tab = tab_option
            
            st.markdown("---")
            st.markdown("### 📋 Application Status")
            
            if tab_option == "Single Prediction":
                if st.session_state.prediction_made:
                    st.info("✅ Single Prediction Complete")
                else:
                    st.warning("⏳ Awaiting Single Application")
            else:
                if st.session_state.bulk_prediction_made:
                    st.info("✅ Bulk Prediction Complete")
                else:
                    st.warning("⏳ Awaiting Bulk Upload")
        else:
            st.info("Please login to continue")
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.markdown("- **Model**: Multilayer Perceptron (MLP)")
        st.markdown("- **Success Rate**: 97.65%")
        st.markdown("- **Features**: 11 input variables")
        st.markdown("- **Accuracy**: Based on 4,269 records")
        st.markdown("---")
        st.markdown("### ℹ️ Description")
        st.markdown("""
            **SmartScore** analyzes credit risk  
            using DL models trained on 4K+ records.  
            It predicts default probability,  
            generates risk scores & scorecards,  
            and provides a predictive dashboard.
            
            **New**: Bulk prediction capability  
            for processing multiple applications  
            with comprehensive analytics.
        """)
                
    # Main content
    if not st.session_state.logged_in:
        login_page()
    else:
        # Handle navigation based on current tab
        if st.session_state.current_tab == "Single Prediction":
            if st.session_state.prediction_made:
                display_prediction_results()
            else:
                loan_application_form()
        else:  # Bulk Prediction
            if st.session_state.bulk_prediction_made:
                bulk_results_dashboard()
            else:
                bulk_prediction_page()

if __name__ == "__main__":
    main()
