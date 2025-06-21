import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 🎨 PROFESSIONAL CSS STYLING ---
st.markdown("""
<style>
/* Import Professional Fonts */
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* CSS Variables for Consistent Design */
:root {
    --primary-color: #2E86AB;
    --secondary-color: #A23B72;
    --accent-color: #F18F01;
    --success-color: #06A77D;
    --warning-color: #F5B800;
    --error-color: #D64545;
    --text-primary: #2C3E50;
    --text-secondary: #5D6D7E;
    --bg-light: #F8F9FA;
    --bg-white: #FFFFFF;
    --border-radius: 12px;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.12);
    --shadow-lg: 0 8px 30px rgba(0, 0, 0, 0.16);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Main App Container */
.stApp {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 50%, #DEE2E6 100%);
    min-height: 100vh;
}

.main .block-container {
    padding: 2rem;
    max-width: 1400px;
    background: var(--bg-white);
    border-radius: 20px;
    box-shadow: var(--shadow-lg);
    margin: 1.5rem auto;
    border: 1px solid rgba(255, 255, 255, 0.9);
}

/* Typography System */
html, body, [class*="css"]:not([class*="icon"]):not([class*="Icon"]), 
[data-testid]:not([data-testid*="icon"]):not([data-testid*="Icon"]), 
.stApp, .main, div:not(button), span:not(button span), p, 
label, .stMarkdown, .stText {
    font-family: 'Kanit', 'Inter', sans-serif !important;
    line-height: 1.6 !important;
    color: var(--text-primary);
}

/* Header Styling */
h1, .stTitle {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 2.8rem !important;
    margin-bottom: 0.5rem !important;
    text-align: center;
    letter-spacing: -0.02em;
}

h2, .stHeader {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 2rem !important;
    margin: 2.5rem 0 1.5rem 0 !important;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid var(--primary-color);
    position: relative;
}

h2::before {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 0;
    width: 60px;
    height: 3px;
    background: var(--accent-color);
}

h3, .stSubheader {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 1.5rem !important;
    margin: 1.8rem 0 1rem 0 !important;
}

/* Professional Card System */
.pro-card {
    background: var(--bg-white);
    padding: 2rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-md);
    margin: 1.5rem 0;
    transition: var(--transition);
    border: 1px solid rgba(46, 134, 171, 0.1);
    position: relative;
    overflow: hidden;
}

.pro-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
}

.pro-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* Status Cards */
.status-card {
    background: linear-gradient(135deg, rgba(46, 134, 171, 0.05), rgba(6, 167, 125, 0.05));
    border-left: 4px solid var(--success-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
}

.warning-card {
    background: linear-gradient(135deg, rgba(245, 184, 0, 0.05), rgba(241, 143, 1, 0.05));
    border-left: 4px solid var(--warning-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
}

.error-card {
    background: linear-gradient(135deg, rgba(214, 69, 69, 0.05), rgba(162, 59, 114, 0.05));
    border-left: 4px solid var(--error-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
}

/* Enhanced Metrics */
.stMetric {
    background: linear-gradient(135deg, var(--bg-white) 0%, var(--bg-light) 100%);
    padding: 1.8rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: var(--transition);
    border: 1px solid rgba(46, 134, 171, 0.1);
    position: relative;
}

.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary-color);
}

.stMetric [data-testid="metric-container"] > div > div:first-child {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.stMetric [data-testid="metric-container"] > div > div:nth-child(2) {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--primary-color) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Professional Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
    background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)) !important;
}

/* Form Elements */
.stSelectbox, .stMultiselect, .stTextInput, .stNumberInput {
    margin: 0.75rem 0;
}

.stSelectbox > div > div, .stMultiselect > div > div, 
.stTextInput > div > div, .stNumberInput > div > div {
    border-radius: var(--border-radius) !important;
    border: 2px solid rgba(46, 134, 171, 0.2) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: var(--transition) !important;
}

.stSelectbox > div > div:focus-within, .stMultiselect > div > div:focus-within,
.stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.1) !important;
}

/* Radio Button Enhancement */
.stRadio > div {
    background: var(--bg-white);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-sm);
    border: 1px solid rgba(46, 134, 171, 0.1);
}

.stRadio > div > label {
    background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-white) 100%);
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    border-radius: calc(var(--border-radius) - 2px);
    transition: var(--transition);
    cursor: pointer;
    border: 1px solid rgba(46, 134, 171, 0.1);
    font-weight: 500;
}

.stRadio > div > label:hover {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    transform: translateX(8px);
    box-shadow: var(--shadow-sm);
}

/* Enhanced Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 1.2rem 1.8rem !important;
    font-weight: 600 !important;
    margin: 1.5rem 0 !important;
    box-shadow: var(--shadow-md) !important;
    transition: var(--transition) !important;
    position: relative;
    overflow: hidden;
}

.streamlit-expanderHeader::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 60px;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1));
}

.streamlit-expanderHeader:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg) !important;
}

.streamlit-expanderContent {
    background: var(--bg-white) !important;
    border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
    padding: 2rem !important;
    box-shadow: var(--shadow-md) !important;
    border: 1px solid rgba(46, 134, 171, 0.1) !important;
    border-top: none !important;
}

/* Professional DataFrames */
.stDataFrame {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(46, 134, 171, 0.1);
}

/* Alert System */
.stAlert {
    border-radius: var(--border-radius) !important;
    border: none !important;
    box-shadow: var(--shadow-md) !important;
    padding: 1.5rem !important;
    margin: 1.5rem 0 !important;
    font-weight: 500 !important;
}

.stSuccess {
    background: linear-gradient(135deg, rgba(6, 167, 125, 0.1), rgba(6, 167, 125, 0.05)) !important;
    border-left: 4px solid var(--success-color) !important;
    color: #0C5F4C !important;
}

.stWarning {
    background: linear-gradient(135deg, rgba(245, 184, 0, 0.1), rgba(245, 184, 0, 0.05)) !important;
    border-left: 4px solid var(--warning-color) !important;
    color: #8B6914 !important;
}

.stError {
    background: linear-gradient(135deg, rgba(214, 69, 69, 0.1), rgba(214, 69, 69, 0.05)) !important;
    border-left: 4px solid var(--error-color) !important;
    color: #8B2635 !important;
}

.stInfo {
    background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(46, 134, 171, 0.05)) !important;
    border-left: 4px solid var(--primary-color) !important;
    color: #1E4A5F !important;
}

/* Professional Sidebar */
.css-1d391kg, .css-1aumxhk {
    background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%) !important;
    padding: 2rem 1rem !important;
}

.css-1d391kg h2, .css-1aumxhk h2,
.css-1d391kg h3, .css-1aumxhk h3 {
    color: white !important;
    border-bottom: 2px solid var(--primary-color) !important;
    padding-bottom: 0.5rem !important;
}

.css-1d391kg .stAlert, .css-1aumxhk .stAlert {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
}

/* Chart Enhancements */
.stPlotlyChart {
    background: var(--bg-white);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-md);
    padding: 1rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(46, 134, 171, 0.1);
}

/* File Uploader */
.stFileUploader {
    background: var(--bg-white);
    border: 2px dashed var(--primary-color);
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    transition: var(--transition);
    margin: 1rem 0;
}

.stFileUploader:hover {
    border-color: var(--secondary-color);
    background: linear-gradient(135deg, rgba(46, 134, 171, 0.02), rgba(162, 59, 114, 0.02));
}

/* Professional Loading */
.stSpinner {
    border-radius: var(--border-radius);
    padding: 2rem;
    background: var(--bg-white);
    box-shadow: var(--shadow-md);
    text-align: center;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 15px;
    }
    
    h1 {
        font-size: 2.2rem !important;
    }
    
    h2 {
        font-size: 1.6rem !important;
    }
    
    .stMetric {
        padding: 1.2rem;
    }
}

/* Icon fixes */
button, svg, .material-icons, [role="button"] {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="collapsedControl"] {
    font-size: 0 !important;
}

[data-testid="collapsedControl"]::after {
    content: "►" !important;
    font-size: 16px !important;
    font-family: monospace !important;
    color: white !important;
}

/* Animation System */
@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.main .block-container > * {
    animation: slideInUp 0.6s ease-out;
}

/* Professional Footer */
.professional-footer {
    background: linear-gradient(135deg, var(--text-primary), var(--primary-color));
    color: white;
    padding: 2rem;
    border-radius: var(--border-radius);
    text-align: center;
    margin-top: 3rem;
    box-shadow: var(--shadow-lg);
}

.footer-logo {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, var(--accent-color), #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.footer-text {
    font-size: 0.95rem;
    opacity: 0.9;
    line-height: 1.8;
}

.footer-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-color), var(--success-color));
    margin: 1rem auto;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- 📱 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Time Series Forecasting | INCD DOE DDC",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎯 PROFESSIONAL HEADER ---
st.markdown("""
<div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, rgba(46, 134, 171, 0.05), rgba(6, 167, 125, 0.05)); border-radius: 15px; margin-bottom: 2rem; border: 1px solid rgba(46, 134, 171, 0.1);">
    <h1 style="margin-bottom: 0.5rem; color: var(--text-primary);">🐳 การพยากรณ์อนุกรมเวลา</h1>
    <h1 style="font-size: 1.8rem; margin-top: 0; color: var(--text-secondary); font-weight: 500;">(Time Series Forecasting)</h1>
    <div style="width: 80px; height: 3px; background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); margin: 1.5rem auto; border-radius: 2px;"></div>
    <p style="font-size: 1.2rem; color: var(--text-secondary); margin-top: 1.5rem; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.8;">
        เครื่องมือ AI ขั้นสูงที่ช่วยพยากรณ์จำนวนผู้ป่วย เช่น โรคไข้หวัดใหญ่ในสัปดาห์ข้างหน้า<br>
        โดยใช้ <strong style="color: var(--primary-color);">Facebook Prophet</strong> และ <strong style="color: var(--success-color);">External Factors</strong>
    </p>
    <div style="margin-top: 1.5rem;">
        <span style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">PROFESSIONAL VERSION</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 📊 DATA CONNECTION SECTION ---
st.markdown('<h2>📊 เชื่อมต่อข้อมูล</h2>', unsafe_allow_html=True)

# Session state management
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "ตัวอย่าง"
if 'external_factors_enabled' not in st.session_state:
    st.session_state.external_factors_enabled = False

# Professional data source selection
col1, col2, col3 = st.columns(3)

with col1:
    google_sheets_selected = st.button(
        "📊 Google Sheets\n(แนะนำสำหรับ Real-time)", 
        help="เหมาะสำหรับการแชร์และอัปเดตข้อมูลแบบ real-time",
        use_container_width=True
    )

with col2:
    csv_upload_selected = st.button(
        "📁 อัปโหลดไฟล์ CSV\n(ความปลอดภัยสูง)", 
        help="เหมาะสำหรับข้อมูลที่ต้องการความปลอดภัยสูง",
        use_container_width=True
    )

with col3:
    sample_data_selected = st.button(
        "🎯 ข้อมูลตัวอย่าง\n(ทดลองใช้งาน)", 
        help="ข้อมูลจำลองสำหรับทดสอบระบบ",
        use_container_width=True
    )

# Store selection in session state
if google_sheets_selected:
    st.session_state.data_selection = "📊 Google Sheets (แนะนำ)"
elif csv_upload_selected:
    st.session_state.data_selection = "📁 อัปโหลดไฟล์ CSV"
elif sample_data_selected:
    st.session_state.data_selection = "🎯 ข้อมูลตัวอย่าง"

# Use default if not selected
if 'data_selection' not in st.session_state:
    st.session_state.data_selection = "🎯 ข้อมูลตัวอย่าง"

data_source = st.session_state.data_selection

# === METHOD 1: Google Sheets ===
if data_source == "📊 Google Sheets (แนะนำ)":
    st.markdown('<h3>🌐 เชื่อมต่อ Google Sheets</h3>', unsafe_allow_html=True)
    
    with st.expander("📋 วิธีตั้งค่า Google Sheets (คลิกเพื่อดู)", expanded=False):
        st.markdown("""
        <div class="pro-card">
        <h4 style="color: var(--primary-color);">🚀 ขั้นตอนที่ 1: เตรียม Google Sheets</h4>
        <ol style="line-height: 1.8;">
            <li>เปิด Google Sheets ใหม่: <a href="https://sheets.google.com" target="_blank" style="color: var(--primary-color);">sheets.google.com</a></li>
            <li>ใส่ข้อมูลตามรูปแบบพื้นฐาน:</li>
        </ol>
        
        <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; margin: 1rem 0;">
        A1: end_date    B1: cases    C1: week_num<br>
        A2: 10/01/2021  B2: 125      C2: 1<br>
        A3: 17/01/2021  B3: 134      C3: 2
        </div>
        
        <h4 style="color: var(--success-color);">🌍 ขั้นตอนที่ 2: เพิ่มปัจจัยภายนอก (เสริม)</h4>
        <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; margin: 1rem 0;">
        D1: temperature  E1: humidity  F1: holiday_flag  G1: campaign<br>
        D2: 25.5         E2: 75        F2: 0             G2: 0
        </div>
        
        <h4 style="color: var(--accent-color);">🔗 ขั้นตอนที่ 3: แชร์ Google Sheets</h4>
        <ol style="line-height: 1.8;">
            <li>คลิกปุ่ม "Share" มุมขวาบน</li>
            <li>เปลี่ยน "Restricted" เป็น <strong style="color: var(--primary-color);">"Anyone with the link"</strong></li>
            <li>ตั้งสิทธิ์เป็น <strong style="color: var(--primary-color);">"Viewer"</strong> หรือ <strong>"Editor"</strong></li>
            <li>คลิก "Copy link"</li>
        </ol>
        
        <div class="warning-card" style="margin-top: 1.5rem;">
            <strong>⚠️ หมายเหตุสำคัญ:</strong> ใช้ 'holiday_flag' แทน 'holidays' เพื่อหลีกเลี่ยงปัญหากับ Prophet
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("💡 **ตัวอย่าง Google Sheets URL:**\n`https://docs.google.com/spreadsheets/d/1ABC123.../edit?usp=sharing`")
    
    sheets_url = st.text_input(
        "🔗 URL ของ Google Sheets:",
        placeholder="วาง Google Sheets URL ที่นี่...",
        help="URL ต้องเป็นแบบ public (Anyone with link can view)"
    )
    
    if sheets_url:
        try:
            if "docs.google.com/spreadsheets" in sheets_url:
                if "/d/" in sheets_url:
                    sheet_id = sheets_url.split("/d/")[1].split("/")[0]
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
                    
                    with st.spinner("🔄 กำลังดาวน์โหลดข้อมูลจาก Google Sheets..."):
                        df_sheets = pd.read_csv(csv_url)
                        
                        required_columns = ['end_date', 'cases', 'week_num']
                        missing_columns = [col for col in required_columns if col not in df_sheets.columns]
                        
                        if missing_columns:
                            st.error(f"❌ Google Sheets ขาดคอลัมน์: {', '.join(missing_columns)}")
                        else:
                            # Data cleaning
                            df_sheets['end_date'] = pd.to_datetime(df_sheets['end_date'], format='%d/%m/%Y', errors='coerce')
                            df_sheets['cases'] = pd.to_numeric(df_sheets['cases'], errors='coerce')
                            df_sheets['week_num'] = pd.to_numeric(df_sheets['week_num'], errors='coerce')
                            
                            external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                                           'population_density', 'school_closed', 'tourists']
                            
                            if 'holidays' in df_sheets.columns and 'holiday_flag' not in df_sheets.columns:
                                df_sheets['holiday_flag'] = df_sheets['holidays']
                                df_sheets.drop('holidays', axis=1, inplace=True)
                                st.info("ℹ️ แปลงคอลัมน์ 'holidays' เป็น 'holiday_flag' แล้ว")
                            
                            for col in external_cols:
                                if col in df_sheets.columns:
                                    df_sheets[col] = pd.to_numeric(df_sheets[col], errors='coerce')
                            
                            df_sheets = df_sheets.dropna(subset=required_columns).reset_index(drop=True)
                            
                            if len(df_sheets) > 0:
                                df_sheets = df_sheets.sort_values('end_date').reset_index(drop=True)
                                has_external = any(col in df_sheets.columns for col in external_cols)
                                
                                st.session_state.current_data = df_sheets
                                st.session_state.data_source = "Google Sheets"
                                st.session_state.external_factors_enabled = has_external
                                
                                st.success(f"✅ เชื่อมต่อ Google Sheets สำเร็จ! {len(df_sheets)} สัปดาห์")
                                
                                if has_external:
                                    available_factors = [col for col in external_cols if col in df_sheets.columns]
                                    st.info(f"🌍 พบปัจจัยภายนอก: {', '.join(available_factors)}")
                                
                                # Display basic metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📅 ช่วงเวลา", f"{df_sheets['end_date'].min().strftime('%d/%m/%Y')} - {df_sheets['end_date'].max().strftime('%d/%m/%Y')}")
                                with col2:
                                    st.metric("👥 ผู้ป่วยเฉลี่ย", f"{df_sheets['cases'].mean():.1f}")
                                with col3:
                                    st.metric("📊 จำนวนสัปดาห์", len(df_sheets))
                                
                                if st.button("🔄 รีเฟรชข้อมูลจาก Google Sheets"):
                                    st.rerun()
                            else:
                                st.error("❌ ไม่พบข้อมูลที่ถูกต้องใน Google Sheets")
                else:
                    st.error("❌ URL ไม่ถูกต้อง กรุณาใช้ URL ของ Google Sheets")
            else:
                st.error("❌ กรุณาใส่ URL ของ Google Sheets")
                
        except Exception as e:
            st.error(f"❌ ไม่สามารถเชื่อมต่อ Google Sheets ได้: {str(e)}")
            st.info("💡 **แนวทางแก้ไข:**\n- ตรวจสอบว่า URL ถูกต้อง\n- ตรวจสอบว่าแชร์เป็น 'Anyone with link'\n- ลองรีเฟรชหน้าเว็บ")

# === METHOD 2: CSV Upload ===
elif data_source == "📁 อัปโหลดไฟล์ CSV":
    st.markdown('<h3>📁 อัปโหลดไฟล์ CSV</h3>', unsafe_allow_html=True)
    
    with st.expander("📋 โครงสร้างไฟล์ CSV ที่ต้องการ", expanded=False):
        st.markdown("""
        <div class="pro-card">
        <h4 style="color: var(--primary-color);">📊 คอลัมน์พื้นฐาน (จำเป็น):</h4>
        <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; margin: 1rem 0;">
        end_date,cases,week_num<br>
        07/01/2024,120,1<br>
        14/01/2024,135,2<br>
        21/01/2024,98,3
        </div>
        
        <h4 style="color: var(--success-color);">🌍 คอลัมน์ปัจจัยภายนอก (เสริม):</h4>
        <div style="background: var(--bg-light); padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; margin: 1rem 0;">
        end_date,cases,week_num,temperature,humidity,holiday_flag,campaign<br>
        07/01/2024,120,1,25.5,75,0,0<br>
        14/01/2024,135,2,23.2,82,1,0<br>
        21/01/2024,98,3,28.1,68,0,1
        </div>
        
        <div class="warning-card">
            <strong>⚠️ หมายเหตุ:</strong> ใช้ 'holiday_flag' แทน 'holidays' เพื่อหลีกเลี่ยงปัญหากับ Prophet
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ CSV",
        type=['csv'],
        help="ไฟล์ต้องมีคอลัมน์: end_date, cases, week_num (+ ปัจจัยภายนอกตามต้องการ)"
    )
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            required_columns = ['end_date', 'cases', 'week_num']
            missing_columns = [col for col in required_columns if col not in df_uploaded.columns]
            
            if missing_columns:
                st.error(f"❌ ไฟล์ขาดคอลัมน์: {', '.join(missing_columns)}")
            else:
                # Data processing (same as Google Sheets)
                df_uploaded['end_date'] = pd.to_datetime(df_uploaded['end_date'], format='%d/%m/%Y', errors='coerce')
                df_uploaded['cases'] = pd.to_numeric(df_uploaded['cases'], errors='coerce')
                df_uploaded['week_num'] = pd.to_numeric(df_uploaded['week_num'], errors='coerce')
                
                external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                               'population_density', 'school_closed', 'tourists']
                
                if 'holidays' in df_uploaded.columns and 'holiday_flag' not in df_uploaded.columns:
                    df_uploaded['holiday_flag'] = df_uploaded['holidays']
                    df_uploaded.drop('holidays', axis=1, inplace=True)
                    st.info("ℹ️ แปลงคอลัมน์ 'holidays' เป็น 'holiday_flag' แล้ว")
                
                for col in external_cols:
                    if col in df_uploaded.columns:
                        df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
                
                df_uploaded = df_uploaded.dropna(subset=required_columns).reset_index(drop=True)
                
                if len(df_uploaded) > 0:
                    df_uploaded = df_uploaded.sort_values('end_date').reset_index(drop=True)
                    has_external = any(col in df_uploaded.columns for col in external_cols)
                    
                    st.session_state.current_data = df_uploaded
                    st.session_state.data_source = f"ไฟล์: {uploaded_file.name}"
                    st.session_state.external_factors_enabled = has_external
                    
                    st.success(f"✅ อัปโหลดไฟล์สำเร็จ! {len(df_uploaded)} สัปดาห์")
                    
                    if has_external:
                        available_factors = [col for col in external_cols if col in df_uploaded.columns]
                        st.info(f"🌍 พบปัจจัยภายนอก: {', '.join(available_factors)}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📅 ช่วงเวลา", f"{df_uploaded['end_date'].min().strftime('%d/%m/%Y')} - {df_uploaded['end_date'].max().strftime('%d/%m/%Y')}")
                    with col2:
                        st.metric("👥 ผู้ป่วยเฉลี่ย", f"{df_uploaded['cases'].mean():.1f}")
                    with col3:
                        st.metric("📊 จำนวนสัปดาห์", len(df_uploaded))
                else:
                    st.error("❌ ไม่พบข้อมูลที่ถูกต้องในไฟล์")
                    
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")

# === METHOD 3: Sample Data ===
else:  # Sample data
    st.markdown('<h3>🎯 ใช้ข้อมูลตัวอย่าง</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        basic_sample = st.button(
            "📊 ข้อมูลพื้นฐาน\n(52 สัปดาห์)", 
            help="ข้อมูลจำลองพื้นฐานสำหรับทดสอบ",
            use_container_width=True
        )
    
    with col2:
        advanced_sample = st.button(
            "🌍 ข้อมูลครบถ้วน\n(52 สัปดาห์ + External Factors)", 
            help="ข้อมูลจำลองพร้อมปัจจัยภายนอก",
            use_container_width=True
        )
    
    if basic_sample:
        st.session_state.sample_type = "📊 ข้อมูลพื้นฐาน (52 สัปดาห์)"
    elif advanced_sample:
        st.session_state.sample_type = "🌍 ข้อมูลพร้อมปัจจัยภายนอก (52 สัปดาห์)"
    
    if 'sample_type' not in st.session_state:
        st.session_state.sample_type = "📊 ข้อมูลพื้นฐาน (52 สัปดาห์)"
        
    sample_type = st.session_state.sample_type
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-07', end='2024-12-29', freq='W')
    cases = []
    
    for i, date in enumerate(dates):
        week_of_year = date.isocalendar()[1]
        seasonal = 80 + 30 * np.sin(2 * np.pi * (week_of_year - 10) / 52)
        trend = -0.2 * i
        noise = np.random.normal(0, 8)
        cases.append(max(10, int(seasonal + trend + noise)))
    
    if sample_type == "📊 ข้อมูลพื้นฐาน (52 สัปดาห์)":
        df_sample = pd.DataFrame({
            'end_date': dates,
            'cases': cases,
            'week_num': range(1, len(dates) + 1)
        })
        st.session_state.external_factors_enabled = False
    else:
        # Add external factors
        temperatures = []
        humidities = []
        holiday_flags = []
        campaigns = []
        
        for i, date in enumerate(dates):
            week_of_year = date.isocalendar()[1]
            temp = 26 + 6 * np.sin(2 * np.pi * (week_of_year - 10) / 52) + np.random.normal(0, 2)
            temperatures.append(round(temp, 1))
            
            humidity = 70 + 15 * np.sin(2 * np.pi * (week_of_year - 20) / 52) + np.random.normal(0, 5)
            humidities.append(max(40, min(95, int(humidity))))
            
            holiday = 1 if week_of_year in [1, 2, 13, 14, 31, 32, 52] else 0
            holiday_flags.append(holiday)
            
            campaign = 1 if week_of_year in range(20, 25) or week_of_year in range(45, 50) else 0
            campaigns.append(campaign)
        
        df_sample = pd.DataFrame({
            'end_date': dates,
            'cases': cases,
            'week_num': range(1, len(dates) + 1),
            'temperature': temperatures,
            'humidity': humidities,
            'holiday_flag': holiday_flags,
            'campaign': campaigns,
            'outbreak_index': np.random.uniform(0.1, 0.8, len(dates)).round(2),
            'population_density': [1250] * len(dates),
            'school_closed': [0 if week_of_year not in [14, 15, 16] else 1 for week_of_year in [date.isocalendar()[1] for date in dates]],
            'tourists': np.random.randint(8000, 20000, len(dates))
        })
        st.session_state.external_factors_enabled = True
    
    st.session_state.current_data = df_sample
    st.session_state.data_source = f"ข้อมูลตัวอย่าง: {sample_type.split(' ')[1]}"
    
    if st.session_state.external_factors_enabled:
        st.info("🌍 ข้อมูลตัวอย่างนี้รวมปัจจัยภายนอก: อุณหภูมิ, ความชื้น, วันหยุด, แคมเปญ, ดัชนีการระบาด, ความหนาแน่นประชากร, การปิดโรงเรียน, นักท่องเที่ยว")

# Check if data exists
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    
    # Display current data status
    st.markdown(f"""
    <div class="status-card">
        <div style="display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--success-color); margin-right: 12px;"></div>
            <h4 style="margin: 0; color: var(--text-primary);">🔄 แหล่งข้อมูลปัจจุบัน: <strong>{st.session_state.data_source}</strong></h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data preview
    with st.expander("👁️ ดูตัวอย่างข้อมูล", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
    # Clear data button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ ล้างข้อมูลและเลือกใหม่", use_container_width=True):
            st.session_state.current_data = None
            st.session_state.data_source = "ตัวอย่าง"
            st.session_state.external_factors_enabled = False
            st.rerun()
            
else:
    st.markdown("""
    <div class="error-card" style="text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--error-color); margin-right: 12px;"></div>
            <h4 style="margin: 0; color: var(--text-primary);">❌ ไม่พบข้อมูล กรุณาเลือกแหล่งข้อมูลด้านบน</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Continue with the rest of the code from the original...
# === DATA QUALITY ANALYSIS ===
st.markdown('<h2>📊 วิเคราะห์คุณภาพข้อมูล</h2>', unsafe_allow_html=True)

# Basic metrics display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 จำนวนสัปดาห์", len(df))
with col2:
    st.metric("📅 ช่วงเวลา", f"{df['week_num'].min()}-{df['week_num'].max()}")
with col3:
    st.metric("🏥 ผู้ป่วยเฉลี่ย", f"{df['cases'].mean():.1f} ราย")
with col4:
    st.metric("⏱️ ช่วงข้อมูล", f"{(df['end_date'].max() - df['end_date'].min()).days // 7} สัปดาห์")

# Data quality assessment
data_quality_score = 100
data_quality_issues = []

# Check data sufficiency
if len(df) < 8:
    data_quality_score -= 30
    data_quality_issues.append({
        'type': 'insufficient_data',
        'severity': 'warning',
        'message': f"ข้อมูลมีเพียง {len(df)} สัปดาห์ (ควรมีอย่างน้อย 8 สัปดาห์)",
        'suggestion': "เพิ่มข้อมูลย้อนหลังให้มากขึ้นเพื่อเพิ่มความแม่นยำ"
    })

# Check outliers
Q1 = df['cases'].quantile(0.25)
Q3 = df['cases'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['cases'] < lower_bound) | (df['cases'] > upper_bound)]
if len(outliers) > 0:
    data_quality_score -= min(20, len(outliers) * 5)
    data_quality_issues.append({
        'type': 'outliers',
        'severity': 'warning',
        'message': f"พบค่าผิดปกติ {len(outliers)} จุด",
        'details': outliers[['week_num', 'end_date', 'cases']].copy(),
        'suggestion': f"ตรวจสอบข้อมูลในสัปดาห์ที่ {', '.join(map(str, outliers['week_num'].values))} - อาจเป็นช่วงระบาดหรือข้อมูลผิดพลาด"
    })

# Check zero/negative values
zero_negative = df[df['cases'] <= 0]
if len(zero_negative) > 0:
    data_quality_score -= 50
    data_quality_issues.append({
        'type': 'zero_negative',
        'severity': 'error',
        'message': f"พบค่าศูนย์หรือติดลบ {len(zero_negative)} จุด",
        'details': zero_negative[['week_num', 'end_date', 'cases']].copy(),
        'suggestion': "แก้ไขให้เป็นค่าบวก หรือใช้ค่าเฉลี่ยของสัปดาห์ข้างเคียง"
    })

# Display quality score
if data_quality_score >= 90:
    quality_color = "var(--success-color)"
    quality_status = "ดีเยี่ยม"
    quality_icon = "🏆"
elif data_quality_score >= 70:
    quality_color = "var(--warning-color)"
    quality_status = "ดี"
    quality_icon = "✅"
elif data_quality_score >= 50:
    quality_color = "var(--error-color)"
    quality_status = "พอใช้"
    quality_icon = "⚠️"
else:
    quality_color = "#8b0000"
    quality_status = "ต้องปรับปรุง"
    quality_icon = "❌"

st.markdown(f"""
<div class="pro-card" style="text-align: center; background: linear-gradient(135deg, rgba(6, 167, 125, 0.05), rgba(46, 134, 171, 0.05));">
    <h3 style="color: {quality_color}; margin-bottom: 1rem;">{quality_icon} คะแนนคุณภาพข้อมูล: {data_quality_score}/100</h3>
    <h4 style="color: {quality_color};">สถานะ: {quality_status}</h4>
    <div class="footer-divider" style="background: {quality_color}; margin: 1rem auto;"></div>
    <p style="color: var(--text-secondary); margin: 0;">ข้อมูลมีคุณภาพ{'สูง' if data_quality_score >= 80 else 'ปานกลาง' if data_quality_score >= 60 else 'ต่ำ'} และ{'เหมาะสม' if data_quality_score >= 70 else 'ต้องปรับปรุง'}สำหรับการพยากรณ์</p>
</div>
""", unsafe_allow_html=True)

# Display quality issues if any
if data_quality_issues:
    st.markdown('<h3>⚠️ รายงานคุณภาพข้อมูล</h3>', unsafe_allow_html=True)
    
    errors = [issue for issue in data_quality_issues if issue['severity'] == 'error']
    warnings = [issue for issue in data_quality_issues if issue['severity'] == 'warning']
    
    for issue in errors + warnings:
        card_class = "error-card" if issue['severity'] == 'error' else "warning-card"
        icon = "❌" if issue['severity'] == 'error' else "⚠️"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="margin-right: 8px;">{icon}</span>
                <h4 style="margin: 0; color: var(--text-primary);">{issue['message']}</h4>
            </div>
            <p style="margin: 0;"><strong>💡 คำแนะนำ:</strong> {issue['suggestion']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'details' in issue and isinstance(issue['details'], pd.DataFrame):
            if issue['type'] == 'outliers':
                details_with_stats = issue['details'].copy()
                details_with_stats['สถิติ'] = details_with_stats['cases'].apply(
                    lambda x: f"{'🔺 สูงกว่าปกติ' if x > upper_bound else '🔻 ต่ำกว่าปกติ'} ({x:.0f} vs ปกติ {lower_bound:.0f}-{upper_bound:.0f})"
                )
                st.dataframe(details_with_stats, use_container_width=True)
            else:
                st.dataframe(issue['details'], use_container_width=True)

# Data distribution visualization
if len(df) > 0:
    st.markdown('<h2>📈 การกระจายและแนวโน้มของข้อมูล</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df['cases'],
            name='จำนวนผู้ป่วย',
            boxpoints='outliers',
            marker_color='var(--primary-color)',
            line_color='var(--secondary-color)'
        ))
        
        fig_box.add_hline(
            y=df['cases'].mean(), 
            line_dash="dash", 
            line_color="var(--accent-color)",
            annotation_text=f"ค่าเฉลี่ย: {df['cases'].mean():.1f}",
            annotation_position="top right"
        )
        
        fig_box.update_layout(
            title="Box Plot: การกระจายของข้อมูล",
            yaxis_title="จำนวนผู้ป่วย (ราย)",
            height=400,
            font=dict(family="Kanit, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Histogram
        fig_hist = px.histogram(
            df, 
            x='cases', 
            nbins=min(20, len(df)//2),
            title="Histogram: การแจกแจงของข้อมูล",
            labels={'cases': 'จำนวนผู้ป่วย (ราย)', 'count': 'ความถี่'},
            color_discrete_sequence=['var(--primary-color)']
        )
        fig_hist.update_layout(
            height=400,
            font=dict(family="Kanit, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# External factors setup
if st.session_state.external_factors_enabled:
    st.markdown('<h2>🌍 การตั้งค่าปัจจัยภายนอก (External Factors)</h2>', unsafe_allow_html=True)
    
    available_factors = []
    external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                    'population_density', 'school_closed', 'tourists']
    
    for col in external_cols:
        if col in df.columns and not df[col].isna().all():
            available_factors.append(col)
    
    if available_factors:
        st.markdown("### 🎯 เลือกปัจจัยที่ต้องการใช้:")
        selected_factors = st.multiselect(
            "ปัจจัยภายนอก:",
            available_factors,
            default=available_factors,
            help="ปัจจัยที่เลือกจะถูกรวมเข้าในโมเดล Prophet"
        )
        
        if selected_factors:
            st.success(f"✅ จะใช้ปัจจัยภายนอก: {', '.join(selected_factors)}")
            
            st.markdown("### 📊 สถิติปัจจัยภายนอก:")
            factor_stats = df[selected_factors].describe().round(2)
            st.dataframe(factor_stats, use_container_width=True)
            
            st.markdown("### 🔮 การตั้งค่าปัจจัยภายนอกสำหรับการพยากรณ์:")
            
            future_factors = {}
            
            for factor in selected_factors:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"**{factor}:**")
                
                with col2:
                    method = st.selectbox(
                        f"วิธีกำหนดค่า {factor}",
                        ["ใช้ค่าเฉลี่ย", "ใช้ค่าล่าสุด", "กำหนดเอง"],
                        key=f"method_{factor}"
                    )
                
                with col3:
                    if method == "ใช้ค่าเฉลี่ย":
                        value = df[factor].mean()
                        st.markdown(f"ค่าเฉลี่ย: **{value:.2f}**")
                    elif method == "ใช้ค่าล่าสุด":
                        value = df[factor].iloc[-1]
                        st.markdown(f"ค่าล่าสุด: **{value:.2f}**")
                    else:
                        value = st.number_input(
                            f"ค่า {factor}",
                            value=float(df[factor].mean()),
                            key=f"custom_{factor}"
                        )
                
                future_factors[factor] = value
        else:
            st.warning("⚠️ ไม่ได้เลือกปัจจัยภายนอกใดๆ - จะใช้โมเดลพื้นฐาน")
            selected_factors = []
    else:
        st.info("ℹ️ ไม่พบปัจจัยภายนอกในข้อมูล - จะใช้โมเดลพื้นฐาน")
        selected_factors = []
else:
    selected_factors = []

# Prepare data for Prophet
prophet_df = pd.DataFrame({
    'ds': df['end_date'],
    'y': df['cases']
})

for factor in selected_factors:
    prophet_df[factor] = df[factor]

prophet_df['week_num'] = df['week_num']

# Prophet reserved names validation
def get_prophet_reserved_names():
    return [
        'ds', 'y', 't', 'trend', 'seasonal', 'seasonality', 
        'holidays', 'holiday', 'mcmc_samples', 'uncertainty_samples',
        'yhat', 'yhat_lower', 'yhat_upper', 'cap', 'floor',
        'additive_terms', 'multiplicative_terms', 'extra_regressors'
    ]

def validate_regressor_names(factors):
    reserved_names = get_prophet_reserved_names()
    invalid_names = [factor for factor in factors if factor in reserved_names]
    
    if invalid_names:
        return False, invalid_names
    return True, []

if selected_factors:
    is_valid, invalid_names = validate_regressor_names(selected_factors)
    if not is_valid:
        st.error(f"❌ **ชื่อปัจจัยต่อไปนี้เป็น reserved names ของ Prophet:** {', '.join(invalid_names)}")
        st.info("💡 **แนวทางแก้ไข:** เปลี่ยนชื่อคอลัมน์ดังกล่าวในไฟล์ข้อมูลเป็นชื่ออื่น เช่น:")
        for name in invalid_names:
            if name == 'holidays':
                st.write(f"- `{name}` → `holiday_flag` หรือ `is_holiday`")
            else:
                st.write(f"- `{name}` → `{name}_factor` หรือ `ext_{name}`")
        st.stop()

# Model training function
def train_prophet_model_with_factors(data, factors):
    split_point = int(len(data) * 0.8)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(data) >= 52 else False,
        seasonality_mode='additive',
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    factor_configs = {
        'temperature': {'prior_scale': 0.5, 'mode': 'additive'},
        'humidity': {'prior_scale': 0.3, 'mode': 'additive'},
        'holiday_flag': {'prior_scale': 1.0, 'mode': 'additive'},
        'campaign': {'prior_scale': 0.8, 'mode': 'multiplicative'},
        'outbreak_index': {'prior_scale': 1.5, 'mode': 'multiplicative'},
        'population_density': {'prior_scale': 0.1, 'mode': 'additive'},
        'school_closed': {'prior_scale': 0.7, 'mode': 'additive'},
        'tourists': {'prior_scale': 0.4, 'mode': 'additive'}
    }
    
    for factor in factors:
        if factor in factor_configs:
            config = factor_configs[factor]
            model.add_regressor(factor, prior_scale=config['prior_scale'], mode=config['mode'])
        else:
            model.add_regressor(factor, prior_scale=0.5, mode='additive')
    
    model.fit(train_data)
    
    if len(test_data) > 0:
        future_test = model.make_future_dataframe(periods=len(test_data), freq='W')
        
        for factor in factors:
            if factor in test_data.columns:
                future_test[factor] = list(train_data[factor]) + list(test_data[factor])
        
        forecast_test = model.predict(future_test)
        
        test_actual = test_data['y'].values
        test_predicted = forecast_test.iloc[-len(test_data):]['yhat'].values
        
        validation_mae = mean_absolute_error(test_actual, test_predicted)
        validation_mape = np.mean(np.abs((test_actual - test_predicted) / test_actual)) * 100
        
        model_final = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True if len(data) >= 52 else False,
            seasonality_mode='additive',
            interval_width=0.95,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        for factor in factors:
            if factor in factor_configs:
                config = factor_configs[factor]
                model_final.add_regressor(factor, prior_scale=config['prior_scale'], mode=config['mode'])
            else:
                model_final.add_regressor(factor, prior_scale=0.5, mode='additive')
        
        model_final.fit(data)
        
        return model_final, validation_mae, validation_mape, True
    else:
        return model, None, None, False

# Train model
st.markdown('<h2>🤖 การเทรนโมเดล AI</h2>', unsafe_allow_html=True)

with st.spinner("🔄 กำลังเทรนโมเดล Prophet..."):
    model, val_mae, val_mape, has_validation = train_prophet_model_with_factors(prophet_df, selected_factors)

st.success("✅ เทรนโมเดลสำเร็จ!")

# Forecasting section
st.markdown('<h2>🔮 พยากรณ์จำนวนผู้ป่วย</h2>', unsafe_allow_html=True)

max_forecast_weeks = min(12, len(df) // 2)

weeks_to_forecast = st.slider(
    "เลือกจำนวนสัปดาห์ที่ต้องการพยากรณ์ไปข้างหน้า:",
    min_value=1,
    max_value=max_forecast_weeks,
    value=min(4, max_forecast_weeks),
    help=f"แนะนำไม่เกิน {max_forecast_weeks} สัปดาห์เพื่อความแม่นยำ"
)

if weeks_to_forecast > len(df) // 4:
    st.warning(f"⚠️ การพยากรณ์ {weeks_to_forecast} สัปดาห์ อาจไม่แม่นยำเนื่องจากข้อมูลจำกัด")

# Generate forecast
future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')

for factor in selected_factors:
    if factor in future_factors:
        historical_values = list(prophet_df[factor])
        future_values = [future_factors[factor]] * weeks_to_forecast
        future[factor] = historical_values + future_values

with st.spinner("🔮 กำลังพยากรณ์..."):
    forecast = model.predict(future)

forecast_future = forecast.tail(weeks_to_forecast).copy()

last_week_num = df['week_num'].max()
forecast_future['week_num'] = range(last_week_num + 1, last_week_num + weeks_to_forecast + 1)

recent_avg = df['cases'].tail(min(4, len(df))).mean()
baseline_forecast = [recent_avg] * weeks_to_forecast

# Forecast results display
st.markdown('<h3>📋 ผลการพยากรณ์</h3>', unsafe_allow_html=True)

if has_validation:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Validation MAE", f"{val_mae:.2f}")
    with col2:
        st.metric("📊 Validation MAPE", f"{val_mape:.1f}%")
    with col3:
        if selected_factors:
            st.metric("🌍 External Factors", f"{len(selected_factors)} ตัว")
        else:
            st.metric("🤖 โมเดล", "พื้นฐาน")

# Forecast table
forecast_display_data = {
    'สัปดาห์ที่': forecast_future['week_num'].astype(int),
    '📅 วันที่': forecast_future['ds'].dt.strftime('%d/%m/%Y'),
    '🤖 Prophet พยากรณ์ (ราย)': forecast_future['yhat'].round(0).astype(int),
    '📊 Baseline เฉลี่ย (ราย)': [int(recent_avg)] * weeks_to_forecast,
    '📈 ต่างจาก Baseline': (forecast_future['yhat'] - recent_avg).round(0).astype(int),
    '📉 ช่วงต่ำ (95% CI)': forecast_future['yhat_lower'].round(0).astype(int),
    '📈 ช่วงสูง (95% CI)': forecast_future['yhat_upper'].round(0).astype(int)
}

if selected_factors:
    for factor in selected_factors:
        if factor in future_factors:
            forecast_display_data[f'🌍 {factor}'] = [future_factors[factor]] * weeks_to_forecast

forecast_display = pd.DataFrame(forecast_display_data)
st.dataframe(forecast_display, use_container_width=True)

# Professional Footer
st.markdown("""
<div class="professional-footer">
    <div class="footer-logo">🚀 Developed with ❤️</div>
    <div class="footer-text">
        <strong>Powered by Facebook Prophet & Streamlit</strong><br>
        Advanced Time Series Forecasting Platform<br>
        <strong>INCD Team DOE, DDC</strong><br>
        Institute for National Capacity Development
    </div>
    <div class="footer-divider"></div>
    <div style="font-size: 0.85rem; opacity: 0.8;">
        Professional Version 2.0 | © 2024 INCD Team
    </div>
</div>
""", unsafe_allow_html=True)

# Professional Sidebar
st.sidebar.markdown('<h2 style="color: white;">📖 ข้อมูลโมเดล</h2>', unsafe_allow_html=True)
st.sidebar.info("""
**🤖 Facebook Prophet Features:**
- ตรวจจับ seasonality อัตโนมัติ
- รองรับ holiday effects
- มี confidence intervals
- ทนทานต่อ missing data
- ตรวจจับการเปลี่ยนแปลง trend
- รองรับ external regressors
""")

st.sidebar.markdown('<h2 style="color: white;">🌍 ปัจจัยภายนอกที่รองรับ</h2>', unsafe_allow_html=True)
st.sidebar.info("""
**ปัจจัยที่สามารถเพิ่มได้:**

🌡️ **อุณหภูมิ** - ไข้หวัดระบาดในอากาศเย็น

💧 **ความชื้น** - ความชื้นต่ำเพิ่มการแพร่เชื้อ

🏥 **วันหยุด** - วันหยุดยาวเพิ่มการเดินทาง

📢 **แคมเปญ** - การรณรงค์ลดการแพร่เชื้อ

🦠 **ดัชนีระบาด** - สถานการณ์ในพื้นที่ใกล้เคียง

👥 **ความหนาแน่น** - พื้นที่แออัดแพร่เชื้อเร็ว

🏫 **โรงเรียน** - การเปิด-ปิดโรงเรียน

✈️ **นักท่องเที่ยว** - การเคลื่อนย้ายคน
""")

st.sidebar.markdown('<h2 style="color: white;">📊 แหล่งข้อมูลปัจจุบัน</h2>', unsafe_allow_html=True)
st.sidebar.info(f"**ใช้ข้อมูลจาก:** {st.session_state.data_source}")

if st.session_state.data_source == "Google Sheets":
    st.sidebar.success("✅ เชื่อมต่อ Google Sheets สำเร็จ - ข้อมูลจะอัปเดตแบบ real-time")
elif "ไฟล์:" in st.session_state.data_source:
    st.sidebar.info("📁 ใช้ไฟล์ที่อัปโหลด - ข้อมูลคงที่ตามไฟล์")
else:
    st.sidebar.warning("🎯 ใช้ข้อมูลตัวอย่าง - เพื่อการทดสอบเท่านั้น")

if st.session_state.external_factors_enabled:
    st.sidebar.success(f"🌍 ใช้ปัจจัยภายนอก: {len(selected_factors)} ตัว")
else:
    st.sidebar.info("📊 ใช้โมเดลพื้นฐาน (ไม่มีปัจจัยภายนอก)")

# Team info
st.sidebar.markdown('<hr style="border-color: rgba(255,255,255,0.3);">', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.6;">
    <p><strong>🏢 INCD Team DOE, DDC</strong></p>
    <p>Institute for National Capacity Development</p>
    <p style="font-size: 0.8rem; opacity: 0.8;">Department of Disease Control</p>
    <p style="font-size: 0.8rem; opacity: 0.8;">Ministry of Public Health</p>
</div>
""", unsafe_allow_html=True)
