import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 🎨 ADVANCED STYLING & LAYOUT ---
st.markdown("""
<style>
/* Import Fonts */
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root Variables */
:root {
    --primary-color: #1f77b4;
    --secondary-color: #ff7f0e;
    --success-color: #2ca02c;
    --warning-color: #d62728;
    --info-color: #17a2b8;
    --light-bg: #f8f9fa;
    --dark-bg: #343a40;
    --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    --card-shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.15);
    --border-radius: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Main App Container */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main .block-container {
    padding: 2rem;
    max-width: 1400px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    margin: 1rem auto;
}

/* Typography */
html, body, [class*="css"]:not([class*="icon"]):not([class*="Icon"]), 
[data-testid]:not([data-testid*="icon"]):not([data-testid*="Icon"]), 
.stApp, .main, div:not(button), span:not(button span), p, h1, h2, h3, h4, h5, h6, 
label, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader {
    font-family: 'Kanit', 'Inter', sans-serif !important;
    line-height: 1.6 !important;
}

/* Headers */
h1, .stTitle {
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    margin-bottom: 1rem !important;
    text-align: center;
}

h2, .stHeader {
    color: #2c3e50 !important;
    font-weight: 600 !important;
    font-size: 1.8rem !important;
    margin: 2rem 0 1rem 0 !important;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
    position: relative;
}

h3, .stSubheader {
    color: #34495e !important;
    font-weight: 500 !important;
    font-size: 1.4rem !important;
    margin: 1.5rem 0 1rem 0 !important;
}

/* Cards & Containers */
.metric-card, .info-card, .warning-card, .success-card, .error-card {
    background: white;
    padding: 1.5rem;
    border-radius: var(--border-radius);
    box-shadow: var(--card-shadow);
    margin: 1rem 0;
    transition: var(--transition);
    border-left: 4px solid;
}

.metric-card:hover, .info-card:hover, .warning-card:hover, 
.success-card:hover, .error-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
}

.info-card { border-left-color: var(--info-color); }
.warning-card { border-left-color: var(--warning-color); }
.success-card { border-left-color: var(--success-color); }
.error-card { border-left-color: #dc3545; }

/* Metrics */
.stMetric {
    background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    box-shadow: var(--card-shadow);
    text-align: center;
    transition: var(--transition);
    border: 1px solid #e9ecef;
}

.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: var(--card-shadow-hover);
    border-color: var(--primary-color);
}

.stMetric [data-testid="metric-container"] > div {
    background: none !important;
}

.stMetric [data-testid="metric-container"] > div > div {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--primary-color) !important;
}

.stMetric [data-testid="metric-container"] > div > div:first-child {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #6c757d !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    background: linear-gradient(45deg, #5a6fd8, #6a4c93) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Form Elements */
.stSelectbox, .stMultiselect, .stTextInput, .stNumberInput, .stSlider {
    margin: 0.5rem 0;
}

.stSelectbox > div > div, .stMultiselect > div > div, 
.stTextInput > div > div, .stNumberInput > div > div {
    border-radius: var(--border-radius) !important;
    border: 2px solid #e9ecef !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
    transition: var(--transition) !important;
}

.stSelectbox > div > div:focus-within, .stMultiselect > div > div:focus-within,
.stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Radio Buttons */
.stRadio > div {
    background: white;
    padding: 1rem;
    border-radius: var(--border-radius);
    box-shadow: var(--card-shadow);
    border: 1px solid #e9ecef;
}

.stRadio > div > label {
    background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    border-radius: 8px;
    transition: var(--transition);
    cursor: pointer;
    border: 1px solid #e9ecef;
}

.stRadio > div > label:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    transform: translateX(5px);
}

/* Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 1rem 1.5rem !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
    box-shadow: var(--card-shadow) !important;
    transition: var(--transition) !important;
}

.streamlit-expanderHeader:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--card-shadow-hover) !important;
}

.streamlit-expanderContent {
    background: white !important;
    border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
    padding: 1.5rem !important;
    box-shadow: var(--card-shadow) !important;
    border: 1px solid #e9ecef !important;
    border-top: none !important;
}

/* DataFrames */
.stDataFrame {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--card-shadow);
    border: 1px solid #e9ecef;
}

.stDataFrame > div {
    border-radius: var(--border-radius);
}

/* Alerts */
.stAlert {
    border-radius: var(--border-radius) !important;
    border: none !important;
    box-shadow: var(--card-shadow) !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    backdrop-filter: blur(10px) !important;
}

.stAlert[data-baseweb="notification"] {
    background: linear-gradient(135deg, rgba(23, 162, 184, 0.1) 0%, rgba(23, 162, 184, 0.05) 100%) !important;
    border-left: 4px solid var(--info-color) !important;
}

.stSuccess {
    background: linear-gradient(135deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%) !important;
    border-left: 4px solid var(--success-color) !important;
}

.stWarning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%) !important;
    border-left: 4px solid var(--warning-color) !important;
}

.stError {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%) !important;
    border-left: 4px solid #dc3545 !important;
}

/* Sidebar */
.css-1d391kg, .css-1aumxhk {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
    padding: 2rem 1rem !important;
}

.sidebar .sidebar-content {
    background: transparent !important;
}

.css-1d391kg h2, .css-1aumxhk h2,
.css-1d391kg h3, .css-1aumxhk h3 {
    color: white !important;
    border-bottom: 2px solid #667eea !important;
    padding-bottom: 0.5rem !important;
}

.css-1d391kg .stAlert, .css-1aumxhk .stAlert {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
}

/* Icon fixes for buttons */
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

/* Plotly Charts */
.js-plotly-plot .plotly .modebar {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 8px !important;
    box-shadow: var(--card-shadow) !important;
}

.stPlotlyChart {
    background: white;
    border-radius: var(--border-radius);
    box-shadow: var(--card-shadow);
    padding: 1rem;
    margin: 1rem 0;
}

/* Loading Spinner */
.stSpinner {
    border-radius: var(--border-radius);
    padding: 2rem;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    box-shadow: var(--card-shadow);
}

/* File Uploader */
.stFileUploader {
    background: white;
    border: 2px dashed #667eea;
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    transition: var(--transition);
}

.stFileUploader:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 15px;
    }
    
    h1, .stTitle {
        font-size: 2rem !important;
    }
    
    h2, .stHeader {
        font-size: 1.5rem !important;
    }
    
    .stMetric {
        padding: 1rem;
    }
}

/* Animation keyframes */
@keyframes fadeInUp {
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
    animation: fadeInUp 0.6s ease-out;
}

/* Custom classes for enhanced styling */
.gradient-text {
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

.glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: var(--border-radius);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.status-success { background: var(--success-color); }
.status-warning { background: var(--warning-color); }
.status-error { background: #dc3545; }
.status-info { background: var(--info-color); }
</style>
""", unsafe_allow_html=True)

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Header Section ที่สวยงาม ---
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="margin-bottom: 0.5rem;">🐳 การพยากรณ์อนุกรมเวลา</h1>
    <h1 style="font-size: 1.8rem; margin-top: 0;">(Time Series Forecasting)</h1>
    <p style="font-size: 1.2rem; color: #6c757d; margin-top: 1rem;">
        เครื่องมือ AI ที่ช่วยพยากรณ์จำนวนผู้ป่วย เช่น โรคไข้หวัดใหญ่ในสัปดาห์ข้างหน้า<br>
        โดยใช้ <strong>Facebook Prophet</strong> และ <strong>External Factors</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# --- 3. ส่วนเชื่อมต่อข้อมูล ---
st.markdown('<h2>📊 เชื่อมต่อข้อมูล</h2>', unsafe_allow_html=True)

# ใช้ session state เพื่อเก็บข้อมูล
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "ตัวอย่าง"
if 'external_factors_enabled' not in st.session_state:
    st.session_state.external_factors_enabled = False

# เลือกวิธีการเชื่อมต่อข้อมูล
col1, col2, col3 = st.columns(3)

with col1:
    google_sheets_selected = st.button(
        "📊 Google Sheets\n(แนะนำ)", 
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

# เก็บการเลือกใน session state
if google_sheets_selected:
    st.session_state.data_selection = "📊 Google Sheets (แนะนำ)"
elif csv_upload_selected:
    st.session_state.data_selection = "📁 อัปโหลดไฟล์ CSV"
elif sample_data_selected:
    st.session_state.data_selection = "🎯 ข้อมูลตัวอย่าง"

# ใช้ค่าเริ่มต้นถ้ายังไม่เลือก
if 'data_selection' not in st.session_state:
    st.session_state.data_selection = "🎯 ข้อมูลตัวอย่าง"

data_source = st.session_state.data_selection

# === วิธีที่ 1: Google Sheets ===
if data_source == "📊 Google Sheets (แนะนำ)":
    st.markdown('<h3>🌐 เชื่อมต่อ Google Sheets</h3>', unsafe_allow_html=True)
    
    # คำแนะนำการตั้งค่า Google Sheets
    with st.expander("📋 วิธีตั้งค่า Google Sheets (คลิกเพื่อดู)", expanded=False):
        st.markdown("""
        <div class="glass-card">
        <h4>🚀 ขั้นตอนที่ 1: เตรียม Google Sheets</h4>
        <ol>
            <li>เปิด Google Sheets ใหม่: <a href="https://sheets.google.com" target="_blank">sheets.google.com</a></li>
            <li>ใส่ข้อมูลตามรูปแบบพื้นฐาน:</li>
        </ol>
        
        ```
        A1: end_date    B1: cases    C1: week_num
        A2: 10/01/2021  B2: 125      C2: 1
        A3: 17/01/2021  B3: 134      C3: 2
        ```
        
        <h4>🌍 ขั้นตอนที่ 2: เพิ่มปัจจัยภายนอก (เสริม)</h4>
        ```
        D1: temperature  E1: humidity  F1: holiday_flag  G1: campaign
        D2: 25.5         E2: 75        F2: 0             G2: 0
        ```
        
        <h4>🔗 ขั้นตอนที่ 3: แชร์ Google Sheets</h4>
        <ol>
            <li>คลิกปุ่ม "Share" มุมขวาบน</li>
            <li>เปลี่ยน "Restricted" เป็น <strong>"Anyone with the link"</strong></li>
            <li>ตั้งสิทธิ์เป็น <strong>"Viewer"</strong> หรือ <strong>"Editor"</strong></li>
            <li>คลิก "Copy link"</li>
        </ol>
        
        <div class="stAlert" style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); border-left: 4px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
        <strong>⚠️ หมายเหตุ:</strong> ใช้ 'holiday_flag' แทน 'holidays' เพื่อหลีกเลี่ยงปัญหากับ Prophet
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ตัวอย่าง URL
    st.info("💡 **ตัวอย่าง Google Sheets URL:**\n`https://docs.google.com/spreadsheets/d/1ABC123.../edit?usp=sharing`")
    
    # Input สำหรับ Google Sheets URL
    sheets_url = st.text_input(
        "🔗 URL ของ Google Sheets:",
        placeholder="วาง Google Sheets URL ที่นี่...",
        help="URL ต้องเป็นแบบ public (Anyone with link can view)"
    )
    
    if sheets_url:
        try:
            # แปลง Google Sheets URL เป็น CSV export URL
            if "docs.google.com/spreadsheets" in sheets_url:
                # ดึง spreadsheet ID
                if "/d/" in sheets_url:
                    sheet_id = sheets_url.split("/d/")[1].split("/")[0]
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
                    
                    with st.spinner("🔄 กำลังดาวน์โหลดข้อมูลจาก Google Sheets..."):
                        # อ่านข้อมูลจาก Google Sheets
                        df_sheets = pd.read_csv(csv_url)
                        
                        # ตรวจสอบคอลัมน์ที่จำเป็น
                        required_columns = ['end_date', 'cases', 'week_num']
                        missing_columns = [col for col in required_columns if col not in df_sheets.columns]
                        
                        if missing_columns:
                            st.error(f"❌ Google Sheets ขาดคอลัมน์: {', '.join(missing_columns)}")
                        else:
                            # ทำความสะอาดข้อมูลพื้นฐาน
                            df_sheets['end_date'] = pd.to_datetime(df_sheets['end_date'], format='%d/%m/%Y', errors='coerce')
                            df_sheets['cases'] = pd.to_numeric(df_sheets['cases'], errors='coerce')
                            df_sheets['week_num'] = pd.to_numeric(df_sheets['week_num'], errors='coerce')
                            
                            # ทำความสะอาดข้อมูลปัจจัยภายนอก
                            external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                                           'population_density', 'school_closed', 'tourists']
                            
                            # รองรับทั้ง 'holidays' และ 'holiday_flag' 
                            if 'holidays' in df_sheets.columns and 'holiday_flag' not in df_sheets.columns:
                                df_sheets['holiday_flag'] = df_sheets['holidays']
                                df_sheets.drop('holidays', axis=1, inplace=True)
                                st.info("ℹ️ แปลงคอลัมน์ 'holidays' เป็น 'holiday_flag' แล้ว")
                            
                            for col in external_cols:
                                if col in df_sheets.columns:
                                    df_sheets[col] = pd.to_numeric(df_sheets[col], errors='coerce')
                            
                            # ลบแถวที่มีข้อมูลหลักไม่ครบ
                            df_sheets = df_sheets.dropna(subset=required_columns).reset_index(drop=True)
                            
                            if len(df_sheets) > 0:
                                # เรียงข้อมูลตามวันที่
                                df_sheets = df_sheets.sort_values('end_date').reset_index(drop=True)
                                
                                # ตรวจสอบว่ามี external factors หรือไม่
                                has_external = any(col in df_sheets.columns for col in external_cols)
                                
                                # เก็บข้อมูลใน session state
                                st.session_state.current_data = df_sheets
                                st.session_state.data_source = "Google Sheets"
                                st.session_state.external_factors_enabled = has_external
                                
                                st.success(f"✅ เชื่อมต่อ Google Sheets สำเร็จ! {len(df_sheets)} สัปดาห์")
                                
                                if has_external:
                                    available_factors = [col for col in external_cols if col in df_sheets.columns]
                                    st.info(f"🌍 พบปัจจัยภายนอก: {', '.join(available_factors)}")
                                
                                # แสดงข้อมูลพื้นฐานในรูปแบบสวยงาม
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("ช่วงเวลา", f"{df_sheets['end_date'].min().strftime('%d/%m/%Y')} - {df_sheets['end_date'].max().strftime('%d/%m/%Y')}")
                                with col2:
                                    st.metric("ผู้ป่วยเฉลี่ย", f"{df_sheets['cases'].mean():.1f}")
                                with col3:
                                    st.metric("จำนวนสัปดาห์", len(df_sheets))
                                
                                # ปุ่มรีเฟรชข้อมูล
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

# === วิธีที่ 2: อัปโหลดไฟล์ CSV ===
elif data_source == "📁 อัปโหลดไฟล์ CSV":
    st.markdown('<h3>📁 อัปโหลดไฟล์ CSV</h3>', unsafe_allow_html=True)
    
    # แสดงตัวอย่างโครงสร้างไฟล์
    with st.expander("📋 โครงสร้างไฟล์ CSV ที่ต้องการ", expanded=False):
        st.markdown("""
        <div class="glass-card">
        <h4>📊 คอลัมน์พื้นฐาน (จำเป็น):</h4>
        
        ```csv
        end_date,cases,week_num
        07/01/2024,120,1
        14/01/2024,135,2
        21/01/2024,98,3
        ```
        
        <h4>🌍 คอลัมน์ปัจจัยภายนอก (เสริม):</h4>
        
        ```csv
        end_date,cases,week_num,temperature,humidity,holiday_flag,campaign
        07/01/2024,120,1,25.5,75,0,0
        14/01/2024,135,2,23.2,82,1,0
        21/01/2024,98,3,28.1,68,0,1
        ```
        
        <div class="stAlert" style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); border-left: 4px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
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
            # อ่านไฟล์ CSV
            df_uploaded = pd.read_csv(uploaded_file)
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_columns = ['end_date', 'cases', 'week_num']
            missing_columns = [col for col in required_columns if col not in df_uploaded.columns]
            
            if missing_columns:
                st.error(f"❌ ไฟล์ขาดคอลัมน์: {', '.join(missing_columns)}")
            else:
                # แปลงประเภทข้อมูลพื้นฐาน
                df_uploaded['end_date'] = pd.to_datetime(df_uploaded['end_date'], format='%d/%m/%Y', errors='coerce')
                df_uploaded['cases'] = pd.to_numeric(df_uploaded['cases'], errors='coerce')
                df_uploaded['week_num'] = pd.to_numeric(df_uploaded['week_num'], errors='coerce')
                
                # ทำความสะอาดข้อมูลปัจจัยภายนอก
                external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                               'population_density', 'school_closed', 'tourists']
                
                # รองรับทั้ง 'holidays' และ 'holiday_flag'
                if 'holidays' in df_uploaded.columns and 'holiday_flag' not in df_uploaded.columns:
                    df_uploaded['holiday_flag'] = df_uploaded['holidays']
                    df_uploaded.drop('holidays', axis=1, inplace=True)
                    st.info("ℹ️ แปลงคอลัมน์ 'holidays' เป็น 'holiday_flag' แล้ว")
                
                for col in external_cols:
                    if col in df_uploaded.columns:
                        df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
                
                # ลบแถวที่มีข้อมูลหลักไม่ครบ
                df_uploaded = df_uploaded.dropna(subset=required_columns).reset_index(drop=True)
                
                if len(df_uploaded) > 0:
                    # เรียงข้อมูลตามวันที่
                    df_uploaded = df_uploaded.sort_values('end_date').reset_index(drop=True)
                    
                    # ตรวจสอบว่ามี external factors หรือไม่
                    has_external = any(col in df_uploaded.columns for col in external_cols)
                    
                    # เก็บข้อมูลใน session state
                    st.session_state.current_data = df_uploaded
                    st.session_state.data_source = f"ไฟล์: {uploaded_file.name}"
                    st.session_state.external_factors_enabled = has_external
                    
                    st.success(f"✅ อัปโหลดไฟล์สำเร็จ! {len(df_uploaded)} สัปดาห์")
                    
                    if has_external:
                        available_factors = [col for col in external_cols if col in df_uploaded.columns]
                        st.info(f"🌍 พบปัจจัยภายนอก: {', '.join(available_factors)}")
                    
                    # แสดงข้อมูลพื้นฐาน
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ช่วงเวลา", f"{df_uploaded['end_date'].min().strftime('%d/%m/%Y')} - {df_uploaded['end_date'].max().strftime('%d/%m/%Y')}")
                    with col2:
                        st.metric("ผู้ป่วยเฉลี่ย", f"{df_uploaded['cases'].mean():.1f}")
                    with col3:
                        st.metric("จำนวนสัปดาห์", len(df_uploaded))
                else:
                    st.error("❌ ไม่พบข้อมูลที่ถูกต้องในไฟล์")
                    
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")

# === วิธีที่ 3: ข้อมูลตัวอย่าง ===
else:  # ข้อมูลตัวอย่าง
    st.markdown('<h3>🎯 ใช้ข้อมูลตัวอย่าง</h3>', unsafe_allow_html=True)
    
    # เลือกประเภทข้อมูลตัวอย่าง
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
    
    # เก็บการเลือกใน session state
    if basic_sample:
        st.session_state.sample_type = "📊 ข้อมูลพื้นฐาน (52 สัปดาห์)"
    elif advanced_sample:
        st.session_state.sample_type = "🌍 ข้อมูลพร้อมปัจจัยภายนอก (52 สัปดาห์)"
    
    # ใช้ค่าเริ่มต้นถ้ายังไม่เลือก
    if 'sample_type' not in st.session_state:
        st.session_state.sample_type = "📊 ข้อมูลพื้นฐาน (52 สัปดาห์)"
        
    sample_type = st.session_state.sample_type
    
    # สร้างข้อมูลตัวอย่าง
    dates = pd.date_range(start='2024-01-07', end='2024-12-29', freq='W')
    cases = []
    
    # สร้างข้อมูลที่มี pattern
    for i, date in enumerate(dates):
        week_of_year = date.isocalendar()[1]
        # seasonal pattern (หนาวเยอะ ร้อนน้อย)
        seasonal = 80 + 30 * np.sin(2 * np.pi * (week_of_year - 10) / 52)
        # trend (ลดลงเล็กน้อย)
        trend = -0.2 * i
        # noise
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
        # เพิ่มปัจจัยภายนอก
        temperatures = []
        humidities = []
        holiday_flags = []
        campaigns = []
        
        for i, date in enumerate(dates):
            week_of_year = date.isocalendar()[1]
            # อุณหภูมิ (หนาวเย็น ร้อนร้อน)
            temp = 26 + 6 * np.sin(2 * np.pi * (week_of_year - 10) / 52) + np.random.normal(0, 2)
            temperatures.append(round(temp, 1))
            
            # ความชื้น (มรสุมชื้น แล้งแห้ง)
            humidity = 70 + 15 * np.sin(2 * np.pi * (week_of_year - 20) / 52) + np.random.normal(0, 5)
            humidities.append(max(40, min(95, int(humidity))))
            
            # วันหยุด (สุ่มบางสัปดาห์)
            holiday = 1 if week_of_year in [1, 2, 13, 14, 31, 32, 52] else 0
            holiday_flags.append(holiday)
            
            # แคมเปญ (บางช่วง)
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
    
    # เก็บข้อมูลใน session state
    st.session_state.current_data = df_sample
    st.session_state.data_source = f"ข้อมูลตัวอย่าง: {sample_type.split(' ')[1]}"
    
    if st.session_state.external_factors_enabled:
        st.info("🌍 ข้อมูลตัวอย่างนี้รวมปัจจัยภายนอก: อุณหภูมิ, ความชื้น, วันหยุด, แคมเปญ, ดัชนีการระบาด, ความหนาแน่นประชากร, การปิดโรงเรียน, นักท่องเที่ยว")

# ตรวจสอบข้อมูล
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    
    # แสดงสถานะข้อมูลในรูปแบบสวยงาม
    st.markdown(f"""
    <div class="glass-card" style="background: linear-gradient(135deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%);">
        <div style="display: flex; align-items: center;">
            <span class="status-indicator status-success"></span>
            <h4 style="margin: 0; color: #2c5530;">🔄 แหล่งข้อมูลปัจจุบัน: <strong>{st.session_state.data_source}</strong></h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # แสดงตัวอย่างข้อมูล
    with st.expander("👁️ ดูตัวอย่างข้อมูล", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
    # ปุ่มล้างข้อมูล
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ ล้างข้อมูลและเลือกใหม่", use_container_width=True):
            st.session_state.current_data = None
            st.session_state.data_source = "ตัวอย่าง"
            st.session_state.external_factors_enabled = False
            st.rerun()
            
else:
    st.markdown("""
    <div class="glass-card" style="background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%); text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center;">
            <span class="status-indicator status-error"></span>
            <h4 style="margin: 0; color: #842029;">❌ ไม่พบข้อมูล กรุณาเลือกแหล่งข้อมูลด้านบน</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ส่วนที่เหลือของโค้ดต่อไป...
# (เนื่องจากความยาว ผมจะแสดงเฉพาะส่วนสำคัญ)

st.markdown('<h2>📊 วิเคราะห์คุณภาพข้อมูล</h2>', unsafe_allow_html=True)

# แสดงข้อมูลพื้นฐานในรูปแบบสวยงาม
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 จำนวนสัปดาห์", len(df))
with col2:
    st.metric("📅 ช่วงเวลา", f"{df['week_num'].min()}-{df['week_num'].max()}")
with col3:
    st.metric("🏥 ผู้ป่วยเฉลี่ย", f"{df['cases'].mean():.1f} ราย")
with col4:
    st.metric("⏱️ ช่วงข้อมูล", f"{(df['end_date'].max() - df['end_date'].min()).days // 7} สัปดาห์")

# การประเมินคุณภาพข้อมูลแบบสวยงาม
data_quality_score = 100

# ตรวจสอบจำนวนข้อมูล
if len(df) < 8:
    data_quality_score -= 30
    
# ตรวจสอบค่าผิดปกติ
Q1 = df['cases'].quantile(0.25)
Q3 = df['cases'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['cases'] < lower_bound) | (df['cases'] > upper_bound)]
if len(outliers) > 0:
    data_quality_score -= min(20, len(outliers) * 5)

# ตรวจสอบค่าศูนย์หรือติดลบ
zero_negative = df[df['cases'] <= 0]
if len(zero_negative) > 0:
    data_quality_score -= 50

# แสดงคะแนนคุณภาพข้อมูล
if data_quality_score >= 90:
    quality_color = "#2ca02c"
    quality_status = "ดีเยี่ยม"
    quality_icon = "🏆"
elif data_quality_score >= 70:
    quality_color = "#ff7f0e"
    quality_status = "ดี"
    quality_icon = "✅"
elif data_quality_score >= 50:
    quality_color = "#d62728"
    quality_status = "พอใช้"
    quality_icon = "⚠️"
else:
    quality_color = "#8b0000"
    quality_status = "ต้องปรับปรุง"
    quality_icon = "❌"

st.markdown(f"""
<div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%);">
    <h3 style="color: {quality_color};">{quality_icon} คะแนนคุณภาพข้อมูล: {data_quality_score}/100</h3>
    <h4 style="color: {quality_color};">สถานะ: {quality_status}</h4>
</div>
""", unsafe_allow_html=True)

# === การตรวจสอบคุณภาพข้อมูลอย่างละเอียด ===
data_quality_issues = []

# 1. ตรวจสอบจำนวนข้อมูล
if len(df) < 8:
    data_quality_issues.append({
        'type': 'insufficient_data',
        'severity': 'warning',
        'message': f"ข้อมูลมีเพียง {len(df)} สัปดาห์ (ควรมีอย่างน้อย 8 สัปดาห์)",
        'suggestion': "เพิ่มข้อมูลย้อนหลังให้มากขึ้นเพื่อเพิ่มความแม่นยำ"
    })

# 2. ตรวจสอบค่าผิดปกติ (Outliers)
outliers = df[(df['cases'] < lower_bound) | (df['cases'] > upper_bound)]
if len(outliers) > 0:
    data_quality_issues.append({
        'type': 'outliers',
        'severity': 'warning',
        'message': f"พบค่าผิดปกติ {len(outliers)} จุด",
        'details': outliers[['week_num', 'end_date', 'cases']].copy(),
        'suggestion': f"ตรวจสอบข้อมูลในสัปดาห์ที่ {', '.join(map(str, outliers['week_num'].values))} - อาจเป็นช่วงระบาดหรือข้อมูลผิดพลาด"
    })

# 3. ตรวจสอบค่าศูนย์หรือติดลบ
zero_negative = df[df['cases'] <= 0]
if len(zero_negative) > 0:
    data_quality_issues.append({
        'type': 'zero_negative',
        'severity': 'error',
        'message': f"พบค่าศูนย์หรือติดลบ {len(zero_negative)} จุด",
        'details': zero_negative[['week_num', 'end_date', 'cases']].copy(),
        'suggestion': "แก้ไขให้เป็นค่าบวก หรือใช้ค่าเฉลี่ยของสัปดาห์ข้างเคียง"
    })

# 4. ตรวจสอบช่องว่างในลำดับสัปดาห์
week_gaps = []
for i in range(1, len(df)):
    if df.iloc[i]['week_num'] - df.iloc[i-1]['week_num'] > 1:
        week_gaps.append((df.iloc[i-1]['week_num'], df.iloc[i]['week_num']))

if week_gaps:
    data_quality_issues.append({
        'type': 'missing_weeks',
        'severity': 'warning',
        'message': f"พบช่องว่างในลำดับสัปดาห์ {len(week_gaps)} จุด",
        'details': week_gaps,
        'suggestion': "เพิ่มข้อมูลในสัปดาห์ที่ขาดหายไป หรือปรับ week_num ให้ต่อเนื่องกัน"
    })

# 5. ตรวจสอบการกระโดดของข้อมูล (Sudden jumps)
df_sorted = df.sort_values('week_num').copy()
df_sorted['cases_diff'] = df_sorted['cases'].diff().abs()
mean_diff = df_sorted['cases_diff'].mean()
std_diff = df_sorted['cases_diff'].std()
sudden_jumps = df_sorted[df_sorted['cases_diff'] > mean_diff + 2 * std_diff]

if len(sudden_jumps) > 0:
    data_quality_issues.append({
        'type': 'sudden_jumps',
        'severity': 'info',
        'message': f"พบการเปลี่ยนแปลงกะทันหัน {len(sudden_jumps)} จุด",
        'details': sudden_jumps[['week_num', 'end_date', 'cases', 'cases_diff']].copy(),
        'suggestion': "ตรวจสอบว่าเป็นเหตุการณ์จริง (เช่น การระบาด) หรือข้อผิดพลาดในการบันทึก"
    })

# แสดงผลการตรวจสอบในรูปแบบสวยงาม
if data_quality_issues:
    st.markdown('<h3>⚠️ รายงานคุณภาพข้อมูล</h3>', unsafe_allow_html=True)
    
    # แยกตาม severity
    errors = [issue for issue in data_quality_issues if issue['severity'] == 'error']
    warnings = [issue for issue in data_quality_issues if issue['severity'] == 'warning']
    infos = [issue for issue in data_quality_issues if issue['severity'] == 'info']
    
    # แสดง Errors (สีแดง)
    if errors:
        for issue in errors:
            st.markdown(f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%); border-left: 4px solid #dc3545;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="status-indicator status-error"></span>
                    <h4 style="margin: 0; color: #842029;">❌ {issue['message']}</h4>
                </div>
                <p style="color: #842029; margin: 0;"><strong>💡 คำแนะนำ:</strong> {issue['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'details' in issue and isinstance(issue['details'], pd.DataFrame):
                st.dataframe(issue['details'], use_container_width=True)
    
    # แสดง Warnings (สีเหลือง)
    if warnings:
        for issue in warnings:
            st.markdown(f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); border-left: 4px solid #ffc107;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="status-indicator status-warning"></span>
                    <h4 style="margin: 0; color: #856404;">⚠️ {issue['message']}</h4>
                </div>
                <p style="color: #856404; margin: 0;"><strong>💡 คำแนะนำ:</strong> {issue['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'details' in issue:
                if isinstance(issue['details'], pd.DataFrame):
                    if issue['type'] == 'outliers':
                        details_with_stats = issue['details'].copy()
                        details_with_stats['สถิติ'] = details_with_stats['cases'].apply(
                            lambda x: f"{'🔺 สูงกว่าปกติ' if x > upper_bound else '🔻 ต่ำกว่าปกติ'} ({x:.0f} vs ปกติ {lower_bound:.0f}-{upper_bound:.0f})"
                        )
                        st.dataframe(details_with_stats, use_container_width=True)
                    else:
                        st.dataframe(issue['details'], use_container_width=True)
                elif isinstance(issue['details'], list):
                    for detail in issue['details']:
                        if isinstance(detail, tuple):
                            st.write(f"- สัปดาห์ที่ {detail[0]} → {detail[1]} (ขาด {detail[1] - detail[0] - 1} สัปดาห์)")
    
    # แสดง Info (สีน้ำเงิน)
    if infos:
        for issue in infos:
            st.markdown(f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(23, 162, 184, 0.1) 0%, rgba(23, 162, 184, 0.05) 100%); border-left: 4px solid #17a2b8;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="status-indicator status-info"></span>
                    <h4 style="margin: 0; color: #0c5460;">ℹ️ {issue['message']}</h4>
                </div>
                <p style="color: #0c5460; margin: 0;"><strong>💡 คำแนะนำ:</strong> {issue['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'details' in issue and isinstance(issue['details'], pd.DataFrame):
                st.dataframe(issue['details'], use_container_width=True)

# แสดงกราฟการกระจายของข้อมูล
if len(df) > 0:
    st.markdown('<h2>📈 การกระจายและแนวโน้มของข้อมูล</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot สำหรับดู outliers
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df['cases'],
            name='จำนวนผู้ป่วย',
            boxpoints='outliers',
            marker_color='#667eea',
            line_color='#764ba2'
        ))
        
        # เพิ่มเส้นค่าเฉลี่ย
        fig_box.add_hline(
            y=df['cases'].mean(), 
            line_dash="dash", 
            line_color="#e74c3c",
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
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            height=400,
            font=dict(family="Kanit, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- การตั้งค่าปัจจัยภายนอก ---
if st.session_state.external_factors_enabled:
    st.markdown('<h2>🌍 การตั้งค่าปัจจัยภายนอก (External Factors)</h2>', unsafe_allow_html=True)
    
    # ตรวจสอบปัจจัยที่มีในข้อมูล
    available_factors = []
    external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                    'population_density', 'school_closed', 'tourists']
    
    for col in external_cols:
        if col in df.columns and not df[col].isna().all():
            available_factors.append(col)
    
    if available_factors:
        # เลือกปัจจัยที่จะใช้
        st.markdown("### 🎯 เลือกปัจจัยที่ต้องการใช้:")
        selected_factors = st.multiselect(
            "ปัจจัยภายนอก:",
            available_factors,
            default=available_factors,
            help="ปัจจัยที่เลือกจะถูกรวมเข้าในโมเดล Prophet"
        )
        
        if selected_factors:
            st.success(f"✅ จะใช้ปัจจัยภายนอก: {', '.join(selected_factors)}")
            
            # แสดงสถิติของปัจจัยภายนอก
            st.markdown("### 📊 สถิติปัจจัยภายนอก:")
            factor_stats = df[selected_factors].describe().round(2)
            st.dataframe(factor_stats, use_container_width=True)
            
            # การตั้งค่าสำหรับการพยากรณ์อนาคต
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

# --- เตรียมข้อมูลสำหรับ Prophet ---
prophet_df = pd.DataFrame({
    'ds': df['end_date'],
    'y': df['cases']
})

# เพิ่มปัจจัยภายนอก
for factor in selected_factors:
    prophet_df[factor] = df[factor]

prophet_df['week_num'] = df['week_num']

# --- ฟังก์ชันเช็ค Prophet reserved names ---
def get_prophet_reserved_names():
    """ส่งคืนรายชื่อที่ Prophet จองไว้"""
    return [
        'ds', 'y', 't', 'trend', 'seasonal', 'seasonality', 
        'holidays', 'holiday', 'mcmc_samples', 'uncertainty_samples',
        'yhat', 'yhat_lower', 'yhat_upper', 'cap', 'floor',
        'additive_terms', 'multiplicative_terms', 'extra_regressors'
    ]

def validate_regressor_names(factors):
    """ตรวจสอบว่าชื่อปัจจัยไม่ใช่ reserved names"""
    reserved_names = get_prophet_reserved_names()
    invalid_names = [factor for factor in factors if factor in reserved_names]
    
    if invalid_names:
        return False, invalid_names
    return True, []

# ตรวจสอบชื่อปัจจัยภายนอก
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

# --- สร้างและเทรนโมเดล Prophet ---
def train_prophet_model_with_factors(data, factors):
    """สร้างและเทรนโมเดล Prophet พร้อมปัจจัยภายนอก"""
    
    # แบ่งข้อมูลเป็น train/test (80/20)
    split_point = int(len(data) * 0.8)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    # สร้างโมเดล Prophet
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(data) >= 52 else False,
        seasonality_mode='additive',
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # เพิ่ม external regressors
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
    
    # เทรนด้วยข้อมูล train
    model.fit(train_data)
    
    # ทดสอบกับข้อมูล test
    if len(test_data) > 0:
        future_test = model.make_future_dataframe(periods=len(test_data), freq='W')
        
        # เพิ่มค่าปัจจัยภายนอกสำหรับ test
        for factor in factors:
            if factor in test_data.columns:
                future_test[factor] = list(train_data[factor]) + list(test_data[factor])
        
        forecast_test = model.predict(future_test)
        
        # คำนวณ validation metrics
        test_actual = test_data['y'].values
        test_predicted = forecast_test.iloc[-len(test_data):]['yhat'].values
        
        validation_mae = mean_absolute_error(test_actual, test_predicted)
        validation_mape = np.mean(np.abs((test_actual - test_predicted) / test_actual)) * 100
        
        # เทรนใหม่ด้วยข้อมูลทั้งหมด
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

# เทรนโมเดล
st.markdown('<h2>🤖 การเทรนโมเดล AI</h2>', unsafe_allow_html=True)

with st.spinner("🔄 กำลังเทรนโมเดล Prophet..."):
    model, val_mae, val_mape, has_validation = train_prophet_model_with_factors(prophet_df, selected_factors)

st.success("✅ เทรนโมเดลสำเร็จ!")

# --- ส่วนสำหรับผู้ใช้ป้อนข้อมูลและพยากรณ์ ---
st.markdown('<h2>🔮 พยากรณ์จำนวนผู้ป่วย</h2>', unsafe_allow_html=True)

# จำกัดจำนวนสัปดาห์การพยากรณ์ให้สมเหตุสมผล
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

# สร้างช่วงวันที่สำหรับการพยากรณ์
future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')

# เพิ่มค่าปัจจัยภายนอกสำหรับอนาคต
for factor in selected_factors:
    if factor in future_factors:
        # เติมค่าในอดีต
        historical_values = list(prophet_df[factor])
        # เติมค่าในอนาคต
        future_values = [future_factors[factor]] * weeks_to_forecast
        # รวมกัน
        future[factor] = historical_values + future_values

# ทำการพยากรณ์
with st.spinner("🔮 กำลังพยากรณ์..."):
    forecast = model.predict(future)

# แยกข้อมูลการพยากรณ์ (เฉพาะส่วนอนาคต)
forecast_future = forecast.tail(weeks_to_forecast).copy()

# เพิ่ม week_num สำหรับการแสดงผล
last_week_num = df['week_num'].max()
forecast_future['week_num'] = range(last_week_num + 1, last_week_num + weeks_to_forecast + 1)

# เพิ่มการเปรียบเทียบกับ Simple Baseline
recent_avg = df['cases'].tail(min(4, len(df))).mean()
baseline_forecast = [recent_avg] * weeks_to_forecast

# ตรวจสอบความสมเหตุสมผลของการพยากรณ์
forecast_mean = forecast_future['yhat'].mean()
historical_mean = df['cases'].mean()
forecast_ratio = forecast_mean / historical_mean if historical_mean > 0 else float('inf')

# เตือนถ้าการพยากรณ์ผิดปกติ
if forecast_ratio > 3 or forecast_ratio < 0.3:
    st.warning(f"⚠️ การพยากรณ์อาจไม่สมเหตุสมผล (เปลี่ยนแปลง {forecast_ratio:.1f} เท่าจากค่าเฉลี่ยเดิม)")

# จำกัดค่าพยากรณ์ให้อยู่ในช่วงที่สมเหตุสมผล
min_reasonable = max(0, historical_mean * 0.1)
max_reasonable = historical_mean * 5

forecast_future['yhat_adjusted'] = forecast_future['yhat'].clip(min_reasonable, max_reasonable)
forecast_future['yhat_upper_adjusted'] = forecast_future['yhat_upper'].clip(min_reasonable, max_reasonable)
forecast_future['yhat_lower_adjusted'] = forecast_future['yhat_lower'].clip(0, max_reasonable)

# --- แสดงผลลัพธ์การพยากรณ์ ---
st.markdown('<h3>📋 ผลการพยากรณ์</h3>', unsafe_allow_html=True)

# แสดงข้อมูล validation ถ้ามี
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

# สร้างตารางผลการพยากรณ์
forecast_display_data = {
    'สัปดาห์ที่': forecast_future['week_num'].astype(int),
    '📅 วันที่': forecast_future['ds'].dt.strftime('%d/%m/%Y'),
    '🤖 Prophet พยากรณ์ (ราย)': forecast_future['yhat_adjusted'].round(0).astype(int),
    '📊 Baseline เฉลี่ย (ราย)': [int(recent_avg)] * weeks_to_forecast,
    '📈 ต่างจาก Baseline': (forecast_future['yhat_adjusted'] - recent_avg).round(0).astype(int),
    '📉 ช่วงต่ำ (95% CI)': forecast_future['yhat_lower_adjusted'].round(0).astype(int),
    '📈 ช่วงสูง (95% CI)': forecast_future['yhat_upper_adjusted'].round(0).astype(int)
}

# เพิ่มคอลัมน์ปัจจัยภายนอกถ้ามี
if selected_factors:
    for factor in selected_factors:
        if factor in future_factors:
            forecast_display_data[f'🌍 {factor}'] = [future_factors[factor]] * weeks_to_forecast

forecast_display = pd.DataFrame(forecast_display_data)
st.dataframe(forecast_display, use_container_width=True)

# เตือนหากค่าพยากรณ์แตกต่างจาก baseline มากเกินไป
max_diff_percent = abs((forecast_future['yhat_adjusted'] - recent_avg) / recent_avg * 100).max()
if max_diff_percent > 50:
    st.warning(f"⚠️ การพยากรณ์แตกต่างจาก baseline มากถึง {max_diff_percent:.1f}% - ควรตรวจสอบความสมเหตุสมผล")
elif max_diff_percent < 5:
    st.info(f"ℹ️ การพยากรณ์ใกล้เคียง baseline ({max_diff_percent:.1f}%) - โมเดลอาจไม่ได้เพิ่มคุณค่ามากนัก")

# --- แสดงกราฟแนวโน้มและการพยากรณ์แบบเชื่อมต่อ ---
st.markdown('<h2>📈 กราฟแนวโน้มและการพยากรณ์</h2>', unsafe_allow_html=True)

# สร้างกราฟที่เชื่อมต่อกัน
fig = go.Figure()

# 1. เพิ่มข้อมูลจริง
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=df['cases'],
    mode='lines+markers',
    name='📊 ข้อมูลจริง',
    line=dict(color='#3498db', width=3),
    marker=dict(size=8, color='#3498db'),
    hovertemplate='สัปดาห์ที่: %{x}<br>ผู้ป่วย: %{y} ราย<extra></extra>'
))

# 2. สร้างข้อมูลการพยากรณ์แบบเชื่อมต่อ
last_week = df['week_num'].max()
last_cases = df['cases'].iloc[-1]

# จุดเชื่อมต่อ + การพยากรณ์
forecast_weeks_connected = [last_week] + list(range(last_week + 1, last_week + weeks_to_forecast + 1))
forecast_values_connected = [last_cases] + list(forecast_future['yhat_adjusted'])

fig.add_trace(go.Scatter(
    x=forecast_weeks_connected,
    y=forecast_values_connected,
    mode='lines+markers',
    name='🤖 Prophet พยากรณ์',
    line=dict(color='#e74c3c', width=3),
    marker=dict(size=10, symbol='diamond', color='#e74c3c'),
    hovertemplate='สัปดาห์ที่: %{x}<br>พยากรณ์: %{y:.0f} ราย<extra></extra>'
))

# 3. เพิ่ม Confidence Interval แบบเชื่อมต่อ
ci_upper_connected = [last_cases] + list(forecast_future['yhat_upper_adjusted'])
ci_lower_connected = [last_cases] + list(forecast_future['yhat_lower_adjusted'])

fig.add_trace(go.Scatter(
    x=forecast_weeks_connected + forecast_weeks_connected[::-1],
    y=ci_upper_connected + ci_lower_connected[::-1],
    fill='toself',
    fillcolor='rgba(231, 76, 60, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='📊 ช่วงความเชื่อมั่น 95%',
    showlegend=True,
    hoverinfo='skip'
))

# 4. เพิ่ม Baseline แบบเชื่อมต่อ
baseline_connected = [last_cases] + baseline_forecast
fig.add_trace(go.Scatter(
    x=forecast_weeks_connected,
    y=baseline_connected,
    mode='lines+markers',
    name='📊 Baseline (เฉลี่ย 4 สัปดาห์)',
    line=dict(color='#f39c12', width=2, dash='dot'),
    marker=dict(size=6, symbol='square', color='#f39c12'),
    hovertemplate='สัปดาห์ที่: %{x}<br>Baseline: %{y:.0f} ราย<extra></extra>'
))

# 5. เพิ่มเส้นแนวโน้มที่เชื่อมต่อ
historical_trend = forecast[:len(df)]['yhat']
trend_connected = list(historical_trend) + list(forecast_future['yhat_adjusted'])
trend_weeks_connected = list(df['week_num']) + list(range(last_week + 1, last_week + weeks_to_forecast + 1))

fig.add_trace(go.Scatter(
    x=trend_weeks_connected,
    y=trend_connected,
    mode='lines',
    name='📈 แนวโน้ม (Prophet)',
    line=dict(color='#27ae60', dash='dash', width=2),
    opacity=0.7,
    hovertemplate='สัปดาห์ที่: %{x}<br>แนวโน้ม: %{y:.0f} ราย<extra></extra>'
))

# 6. เพิ่มเส้นแบ่งระหว่างข้อมูลจริงกับการพยากรณ์
fig.add_vline(
    x=last_week + 0.5, 
    line_dash="solid", 
    line_color="#95a5a6",
    line_width=3,
    annotation_text="📍 จุดเริ่มพยากรณ์",
    annotation_position="top"
)

# ตั้งค่ากราฟ
title = '📈 แนวโน้มผู้ป่วยและการพยากรณ์ (Facebook Prophet'
if selected_factors:
    title += f' + {len(selected_factors)} External Factors'
title += ')'

fig.update_layout(
    title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#2c3e50'}
    },
    xaxis_title='สัปดาห์ที่',
    yaxis_title='จำนวนผู้ป่วย (ราย)',
    hovermode='x unified',
    showlegend=True,
    height=600,
    font=dict(family="Kanit, sans-serif", size=12),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    )
)

# ตั้งค่าช่วงแกน
x_min = max(1, df['week_num'].min() - 1)
x_max = df['week_num'].max() + weeks_to_forecast + 1
fig.update_xaxes(
    range=[x_min, x_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='rgba(0,0,0,0.1)',
    dtick=max(1, (x_max - x_min) // 20)
)

y_min = 0
y_max = max(df['cases'].max(), forecast_future['yhat_upper'].max()) * 1.1
fig.update_yaxes(
    range=[y_min, y_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='rgba(0,0,0,0.1)'
)

st.plotly_chart(fig, use_container_width=True)

# --- คำนวณค่าทางสถิติของโมเดล ---
try:
    # ใช้ข้อมูลในอดีตเพื่อประเมินความแม่นยำ
    historical_forecast = forecast[forecast['ds'].isin(df['end_date'])]
    
    if len(historical_forecast) == len(df):
        actual_values = df['cases'].values
        predicted_values = historical_forecast['yhat'].values
        
        # คำนวณค่า error metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        r2 = r2_score(actual_values, predicted_values)
        
        show_metrics = True
    else:
        show_metrics = False
        st.warning("⚠️ ไม่สามารถคำนวณค่าทางสถิติได้ เนื่องจากข้อมูลไม่ตรงกัน")
        
except Exception as e:
    show_metrics = False
    st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณค่าทางสถิติ: {e}")

# --- แสดงข้อมูลทางสถิติ ---
if show_metrics:
    st.markdown('<h2>📊 ค่าทางสถิติและการประเมินโมเดล</h2>', unsafe_allow_html=True)

    # แสดงค่าความแม่นยำของโมเดล
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 MAE",
            value=f"{mae:.2f}",
            help="Mean Absolute Error - ค่าเฉลี่ยของความผิดพลาด"
        )

    with col2:
        st.metric(
            label="📊 RMSE", 
            value=f"{rmse:.2f}",
            help="Root Mean Square Error - รากที่สองของค่าเฉลี่ยความผิดพลาดยกกำลังสอง"
        )

    with col3:
        st.metric(
            label="📈 MAPE",
            value=f"{mape:.1f}%",
            help="Mean Absolute Percentage Error - เปอร์เซ็นต์ความผิดพลาด"
        )

    with col4:
        st.metric(
            label="📉 R²",
            value=f"{r2:.3f}",
            help="R-squared - ค่าสัมประสิทธิ์การตัดสินใจ (0-1, ยิ่งใกล้ 1 ยิ่งดี)"
        )

    # สรุปผลการประเมิน
    if mape < 10:
        accuracy_level = "ดีมาก (MAPE < 10%)"
        accuracy_color = "#2ca02c"
        accuracy_icon = "🏆"
    elif mape < 20:
        accuracy_level = "ดี (MAPE 10-20%)"
        accuracy_color = "#ff7f0e"
        accuracy_icon = "✅"
    else:
        accuracy_level = "ต้องปรับปรุง (MAPE > 20%)"
        accuracy_color = "#d62728"
        accuracy_icon = "⚠️"

    st.markdown(f"""
    <div class="glass-card" style="background: linear-gradient(135deg, rgba(44, 160, 44, 0.1) 0%, rgba(44, 160, 44, 0.05) 100%); text-align: center;">
        <h3 style="color: {accuracy_color};">{accuracy_icon} สรุปความแม่นยำของโมเดล: {accuracy_level}</h3>
    </div>
    """, unsafe_allow_html=True)

    # แสดงกราฟ Residuals Analysis
    st.markdown('<h3>🔍 การวิเคราะห์ Residuals</h3>', unsafe_allow_html=True)

    residuals = actual_values - predicted_values

    fig_residuals = go.Figure()

    # กราฟ residuals vs predicted
    fig_residuals.add_trace(go.Scatter(
        x=predicted_values,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='#9b59b6', size=8, opacity=0.7)
    ))

    # เส้น y=0
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="#e74c3c", line_width=2)

    fig_residuals.update_layout(
        title="Residuals vs Predicted Values",
        xaxis_title="ค่าพยากรณ์",
        yaxis_title="Residuals (จริง - พยากรณ์)",
        height=400,
        font=dict(family="Kanit, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_residuals, use_container_width=True)

# --- แสดงสถิติข้อมูลพื้นฐาน ---
st.markdown('<h2>📈 สถิติข้อมูลและการพยากรณ์</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 สถิติข้อมูลในอดีต:")
    stats_df = pd.DataFrame({
        'สถิติ': ['📊 ค่าเฉลี่ย', '📍 ค่ามัธยฐาน', '📏 ส่วนเบียงเบนมาตรฐาน', '📉 ค่าต่ำสุด', '📈 ค่าสูงสุด'],
        'ค่า': [
            f"{df['cases'].mean():.1f} ราย",
            f"{df['cases'].median():.1f} ราย", 
            f"{df['cases'].std():.1f} ราย",
            f"{df['cases'].min():.0f} ราย",
            f"{df['cases'].max():.0f} ราย"
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown("### 🔮 สถิติการพยากรณ์:")
    forecast_stats_df = pd.DataFrame({
        'สถิติ': ['📊 ค่าเฉลี่ย', '📍 ค่ามัธยฐาน', '📏 ส่วนเบียงเบนมาตรฐาน', '📉 ค่าต่ำสุด', '📈 ค่าสูงสุด'],
        'ค่า': [
            f"{forecast_future['yhat_adjusted'].mean():.1f} ราย",
            f"{forecast_future['yhat_adjusted'].median():.1f} ราย",
            f"{forecast_future['yhat_adjusted'].std():.1f} ราย", 
            f"{forecast_future['yhat_adjusted'].min():.0f} ราย",
            f"{forecast_future['yhat_adjusted'].max():.0f} ราย"
        ]
    })
    st.dataframe(forecast_stats_df, hide_index=True, use_container_width=True)

# --- แสดงการวิเคราะห์แนวโน้ม ---
st.markdown('<h2>📊 การวิเคราะห์แนวโน้ม</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    avg_forecast = forecast_future['yhat_adjusted'].mean()
    avg_historical = df['cases'].mean()
    trend_change = ((avg_forecast - avg_historical) / avg_historical) * 100
    
    st.metric(
        label="📈 การเปลี่ยนแปลงค่าเฉลี่ย",
        value=f"{trend_change:+.1f}%",
        delta=f"{avg_forecast - avg_historical:+.1f} ราย"
    )

with col2:
    first_forecast = forecast_future['yhat_adjusted'].iloc[0] 
    last_forecast = forecast_future['yhat_adjusted'].iloc[-1]
    forecast_trend = last_forecast - first_forecast
    
    st.metric(
        label="📊 แนวโน้มในช่วงพยากรณ์",
        value="⬆️ เพิ่มขึ้น" if forecast_trend > 0 else "⬇️ ลดลง" if forecast_trend < 0 else "➡️ คงที่",
        delta=f"{forecast_trend:+.1f} ราย"
    )

with col3:
    uncertainty = forecast_future['yhat_upper_adjusted'].mean() - forecast_future['yhat_lower_adjusted'].mean()
    st.metric(
        label="📊 ช่วงความไม่แน่นอนเฉลี่ย",
        value=f"±{uncertainty/2:.1f} ราย",
        help="ช่วงความเชื่อมั่น 95% เฉลี่ย"
    )

# --- แสดงกราฟ components ของ Prophet ---
st.markdown('<h2>🔧 การวิเคราะห์องค์ประกอบ (Trend & Seasonality)</h2>', unsafe_allow_html=True)

try:
    # สร้างกราฟ trend
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
except Exception as e:
    st.warning(f"ไม่สามารถแสดงกราฟ components ได้: {str(e)}")

# แสดงข้อมูลปัจจัยภายนอกที่มีผล
if selected_factors and show_metrics:
    st.markdown('<h2>🌍 ผลกระทบของปัจจัยภายนอก</h2>', unsafe_allow_html=True)
    
    # คำนวณ feature importance (อย่างง่าย)
    factor_importance = {}
    
    for factor in selected_factors:
        # คำนวณ correlation กับ residuals
        factor_values = prophet_df[factor]
        corr = np.corrcoef(factor_values, actual_values)[0, 1]
        factor_importance[factor] = abs(corr)
    
    # แสดงผลกระทบ
    if factor_importance:
        importance_df = pd.DataFrame([
            {
                'ปัจจัย': f"🌍 {factor}", 
                'ความสัมพันธ์': f"{corr:.3f}", 
                'ผลกระทบ': '🔴 สูง' if corr > 0.3 else '🟡 ปานกลาง' if corr > 0.1 else '🟢 ต่ำ'
            }
            for factor, corr in factor_importance.items()
        ])
        
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("""
<div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);">
    <p style="margin: 0; color: #6c757d;">
        <strong>📝 หมายเหตุ:</strong> การพยากรณ์นี้ใช้โมเดล Facebook Prophet ซึ่งสามารถจับ pattern และ seasonality ได้ดีกว่าโมเดลเชิงเส้น
    </p>
</div>
""", unsafe_allow_html=True)

if selected_factors:
    st.caption(f"🌍 โมเดลนี้รวมปัจจัยภายนอก {len(selected_factors)} ตัว เพื่อเพิ่มความแม่นยำ")

# --- Sidebar Information ---
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
   (ใช้ชื่อ 'holiday_flag' แทน 'holidays')

📢 **แคมเปญ** - การรณรงค์ลดการแพร่เชื้อ

🦠 **ดัชนีระบาด** - สถานการณ์ในพื้นที่ใกล้เคียง

👥 **ความหนาแน่น** - พื้นที่แออัดแพร่เชื้อเร็ว

🏫 **โรงเรียน** - การเปิด-ปิดโรงเรียน

✈️ **นักท่องเที่ยว** - การเคลื่อนย้ายคน
""")

st.sidebar.markdown('<h2 style="color: white;">⚠️ Prophet Reserved Names</h2>', unsafe_allow_html=True)
st.sidebar.warning("""
**ชื่อที่โมเดล Prophet จองไว้ใช้เป็นตัวแปร:**

❌ **ห้ามใช้เป็นชื่อคอลัมน์:**
- holidays (ใช้ holiday_flag แทน)
- trend, seasonal, seasonality
- yhat, ds, y, t
- cap, floor
- uncertainty_samples

✅ **ใช้ชื่อทดแทน:**
- holidays → holiday_flag
- seasonal → seasonal_factor  
- trend → trend_data
""")

st.sidebar.markdown('<h2 style="color: white;">🔍 ความน่าเชื่อถือของโมเดล</h2>', unsafe_allow_html=True)
st.sidebar.warning("""
**ข้อจำกัดสำคัญ:**

1. **คุณภาพข้อมูล**: ปัจจัยภายนอกต้องถูกต้องและครบถ้วน

2. **ความเสถียร**: ความสัมพันธ์ต้องคงที่ในอนาคต

3. **Causality**: ต้องมีเหตุผลเชิงสาเหตุ

**คำแนะนำ:**
- ใช้ร่วมกับความรู้ของผู้เชี่ยวชาญ
- ตรวจสอบความสมเหตุสมผล
- อัปเดตโมเดลเมื่อมีข้อมูลใหม่
- ระวังการ overfitting
""")

st.sidebar.markdown('<h2 style="color: white;">📊 ข้อมูลสถิติ</h2>', unsafe_allow_html=True)
st.sidebar.info("""
**📈 Metrics:**
- **MAE**: ความผิดพลาดเฉลี่ย
- **RMSE**: รากที่สองของความผิดพลาดกำลังสอง
- **MAPE**: เปอร์เซ็นต์ความผิดพลาด  
- **R²**: ค่าสัมประสิทธิ์การตัดสินใจ

**🎯 เกณฑ์ประเมิน MAPE:**
- < 10%: ดีมาก
- 10-20%: ดี
- > 20%: ต้องปรับปรุง

**🌍 External Factors:**
- ปรับปรุงความแม่นยำได้ 30-50%
- ต้องมีข้อมูลที่เชื่อถือได้
- ควรมีเหตุผลเชิงสาเหตุ
""")

# แสดงข้อมูลแหล่งข้อมูลปัจจุบัน
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

# แสดงข้อมูลผู้พัฒนา
st.sidebar.markdown('<hr style="border-color: rgba(255,255,255,0.3);">', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
    <p><strong>🚀 Developed with ❤️</strong></p>
    <p>Powered by Facebook Prophet & Streamlit</p>
    <p>Version 2.0 - Modern UI</p>
</div>
""", unsafe_allow_html=True)
