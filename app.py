import streamlit as st
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid], .stApp, .main, 
div, span, p, h1, h2, h3, h4, h5, h6, label, button, input {
    font-family: 'Kanit', 'Sarabun', 'Prompt', sans-serif !important;
    line-height: 1.6 !important;
}

/* สำหรับ Plotly charts */
.plotly * {
    font-family: 'Kanit', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="🖥️",
    layout="wide"
)

st.title("🐳 การพยากรณ์อนุกรมเวลา (Time Series Forecasting)")
st.write("เครื่องมือนี้ช่วยพยากรณ์จำนวนผู้ป่วย เช่น โรคไข้หวัดใหญ่ในสัปดาห์ข้างหน้า โดยใช้ Facebook Prophet")

# --- 2. การเชื่อมต่อข้อมูลหลายรูปแบบ ---
st.subheader("📊 เชื่อมต่อข้อมูล")

# ใช้ session state เพื่อเก็บข้อมูล
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "ตัวอย่าง"
if 'external_factors_enabled' not in st.session_state:
    st.session_state.external_factors_enabled = False

# เลือกวิธีการเชื่อมต่อข้อมูล
data_source = st.radio(
    "เลือกแหล่งข้อมูล:",
    ["📊 Google Sheets (แนะนำ)", "📁 อัปโหลดไฟล์ CSV", "🎯 ข้อมูลตัวอย่าง"],
    help="Google Sheets เหมาะสำหรับการแชร์และอัปเดตข้อมูลแบบ real-time"
)

# === วิธีที่ 1: Google Sheets ===
if data_source == "📊 Google Sheets (แนะนำ)":
    st.markdown("### 🌐 เชื่อมต่อ Google Sheets")
    
    # คำแนะนำการตั้งค่า Google Sheets
    with st.expander("📋 วิธีตั้งค่า Google Sheets (คลิกเพื่อดู)"):
        st.markdown("""
        **ขั้นตอนที่ 1: เตรียม Google Sheets**
        1. เปิด Google Sheets ใหม่: [sheets.google.com](https://sheets.google.com)
        2. ใส่ข้อมูลตามรูปแบบพื้นฐาน:
           ```
           A1: end_date    B1: cases    C1: week_num
           A2: 10/01/2021  B2: 125      C2: 1
           A3: 17/01/2021  B3: 134      C3: 2
           ```
        3. **(เสริม)** เพิ่มปัจจัยภายนอก:
           ```
           D1: temperature  E1: humidity  F1: holiday_flag  G1: campaign
           D2: 25.5         E2: 75        F2: 0             G2: 0
           ```
        
        **ขั้นตอนที่ 2: แชร์ Google Sheets**
        1. คลิกปุ่ม "Share" มุมขวาบน
        2. เปลี่ยน "Restricted" เป็น **"Anyone with the link"**
        3. ตั้งสิทธิ์เป็น **"Viewer"** หรือ **"Editor"**
        4. คลิก "Copy link"
        
        **หมายเหตุ:** ใช้ 'holiday_flag' แทน 'holidays' เพื่อหลีกเลี่ยงปัญหากับ Prophet
        """)
    
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
                            
                            # ทำความสะอาดข้อมูลปัจจัยภายนอก (แก้ไขชื่อคอลัมน์)
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
                                
                                # แสดงข้อมูลพื้นฐาน
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
    st.markdown("### 📁 อัปโหลดไฟล์ CSV")
    
    # แสดงตัวอย่างโครงสร้างไฟล์
    with st.expander("📋 โครงสร้างไฟล์ CSV ที่ต้องการ"):
        st.markdown("""
        **คอลัมน์พื้นฐาน (จำเป็น):**
        ```csv
        end_date,cases,week_num
        07/01/2024,120,1
        14/01/2024,135,2
        21/01/2024,98,3
        ```
        
        **คอลัมน์ปัจจัยภายนอก (เสริม):**
        ```csv
        end_date,cases,week_num,temperature,humidity,holiday_flag,campaign
        07/01/2024,120,1,25.5,75,0,0
        14/01/2024,135,2,23.2,82,1,0
        21/01/2024,98,3,28.1,68,0,1
        ```
        
        **หมายเหตุ:** ใช้ 'holiday_flag' แทน 'holidays' เพื่อหลีกเลี่ยงปัญหากับ Prophet
        """)
    
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
                
                # ทำความสะอาดข้อมูลปัจจัยภายนอก (แก้ไขชื่อคอลัมน์)
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
    st.markdown("### 🎯 ใช้ข้อมูลตัวอย่าง")
    
    # เลือกประเภทข้อมูลตัวอย่าง
    sample_type = st.radio(
        "เลือกประเภทข้อมูลตัวอย่าง:",
        ["📊 ข้อมูลพื้นฐาน (52 สัปดาห์)", "🌍 ข้อมูลพร้อมปัจจัยภายนอก (52 สัปดาห์)"],
        help="ข้อมูลจำลองสำหรับทดสอบระบบ"
    )
    
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
        holiday_flags = []  # เปลี่ยนจาก holidays เป็น holiday_flags
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
            'holiday_flag': holiday_flags,  # เปลี่ยนจาก holidays
            'campaign': campaigns,
            'outbreak_index': np.random.uniform(0.1, 0.8, len(dates)).round(2),
            'population_density': [1250] * len(dates),
            'school_closed': [0 if week_of_year not in [14, 15, 16] else 1 for week_of_year in [date.isocalendar()[1] for date in dates]],
            'tourists': np.random.randint(8000, 20000, len(dates))
        })
        st.session_state.external_factors_enabled = True
    
    # เก็บข้อมูลใน session state
    st.session_state.current_data = df_sample
    st.session_state.data_source = f"ข้อมูลตัวอย่าง: {sample_type.split(' ')[0][2:]}"
    
    if st.session_state.external_factors_enabled:
        st.info("🌍 ข้อมูลตัวอย่างนี้รวมปัจจัยภายนอก: อุณหภูมิ, ความชื้น, วันหยุด, แคมเปญ, ดัชนีการระบาด, ความหนาแน่นประชากร, การปิดโรงเรียน, นักท่องเที่ยว")

# ใช้ข้อมูลที่เก็บใน session state
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    st.info(f"🔄 แหล่งข้อมูลปัจจุบัน: **{st.session_state.data_source}**")
    
    # แสดงตัวอย่างข้อมูล
    with st.expander("ดูตัวอย่างข้อมูล"):
        st.dataframe(df.head(10))
        
    # ปุ่มล้างข้อมูล
    if st.button("🗑️ ล้างข้อมูลและเลือกใหม่"):
        st.session_state.current_data = None
        st.session_state.data_source = "ตัวอย่าง"
        st.session_state.external_factors_enabled = False
        st.rerun()
else:
    st.error("❌ ไม่พบข้อมูล กรุณาเลือกแหล่งข้อมูลด้านบน")
    st.stop()

# === การตรวจสอบคุณภาพข้อมูลอย่างละเอียด ===
st.subheader("📊 วิเคราะห์คุณภาพข้อมูล")

# แสดงข้อมูลพื้นฐาน
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("จำนวนสัปดาห์", len(df))
with col2:
    st.metric("ช่วงเวลา", f"{df['week_num'].min()}-{df['week_num'].max()}")
with col3:
    st.metric("ผู้ป่วยเฉลี่ย", f"{df['cases'].mean():.1f} ราย")
with col4:
    st.metric("ช่วงข้อมูล", f"{(df['end_date'].max() - df['end_date'].min()).days // 7} สัปดาห์")

# การตรวจสอบคุณภาพข้อมูล
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
Q1 = df['cases'].quantile(0.25)
Q3 = df['cases'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

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

# แสดงผลการตรวจสอบ
if data_quality_issues:
    st.subheader("⚠️ การตรวจสอบคุณภาพข้อมูล")
    
    # แยกตาม severity
    errors = [issue for issue in data_quality_issues if issue['severity'] == 'error']
    warnings = [issue for issue in data_quality_issues if issue['severity'] == 'warning']
    infos = [issue for issue in data_quality_issues if issue['severity'] == 'info']
    
    # แสดง Errors (สีแดง)
    if errors:
        for issue in errors:
            st.error(f"❌ **{issue['message']}**")
            st.write(f"💡 **คำแนะนำ:** {issue['suggestion']}")
            
            if 'details' in issue:
                if isinstance(issue['details'], pd.DataFrame):
                    st.write("📋 **รายละเอียด:**")
                    st.dataframe(issue['details'], use_container_width=True)
                elif isinstance(issue['details'], list):
                    for detail in issue['details']:
                        st.write(f"- {detail}")
            st.write("---")
    
    # แสดง Warnings (สีเหลือง)
    if warnings:
        for issue in warnings:
            st.warning(f"⚠️ **{issue['message']}**")
            st.write(f"💡 **คำแนะนำ:** {issue['suggestion']}")
            
            if 'details' in issue:
                if isinstance(issue['details'], pd.DataFrame):
                    st.write("📋 **รายละเอียดข้อมูลที่ผิดปกติ:**")
                    
                    # ถ้าเป็น outliers ให้แสดงเพิ่มเติม
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
                        else:
                            st.write(f"- {detail}")
            st.write("---")
    
    # แสดง Info (สีน้ำเงิน)
    if infos:
        for issue in infos:
            st.info(f"ℹ️ **{issue['message']}**")
            st.write(f"💡 **คำแนะนำ:** {issue['suggestion']}")
            
            if 'details' in issue:
                if isinstance(issue['details'], pd.DataFrame):
                    st.write("📋 **รายละเอียด:**")
                    st.dataframe(issue['details'], use_container_width=True)
            st.write("---")
    
    # สรุปและคำแนะนำรวม
    if errors:
        st.error("🚨 **พบข้อผิดพลาดที่ต้องแก้ไข** - การพยากรณ์อาจไม่แม่นยำ")
    elif warnings:
        st.warning("⚠️ **พบจุดที่ควรตรวจสอบ** - การพยากรณ์ยังใช้ได้แต่ควรปรับปรุง")
    else:
        st.success("✅ **คุณภาพข้อมูลดี** - พร้อมสำหรับการพยากรณ์")
else:
    st.success("✅ **คุณภาพข้อมูลดีมาก** - ไม่พบข้อผิดพลาดหรือค่าผิดปกติ")

# แสดงกราฟการกระจายของข้อมูล
if len(df) > 0:
    st.subheader("📈 การกระจายและแนวโน้มของข้อมูล")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot สำหรับดู outliers
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df['cases'],
            name='จำนวนผู้ป่วย',
            boxpoints='outliers',
            marker_color='lightblue'
        ))
        
        # เพิ่มเส้นค่าเฉลี่ย
        fig_box.add_hline(
            y=df['cases'].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"ค่าเฉลี่ย: {df['cases'].mean():.1f}"
        )
        
        fig_box.update_layout(
            title="Box Plot: การกระจายของข้อมูล",
            yaxis_title="จำนวนผู้ป่วย (ราย)",
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Histogram
        fig_hist = px.histogram(
            df, 
            x='cases', 
            nbins=min(20, len(df)//2),
            title="Histogram: การแจกแจงของข้อมูล",
            labels={'cases': 'จำนวนผู้ป่วย (ราย)', 'count': 'ความถี่'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

# --- การตั้งค่าปัจจัยภายนอก ---
if st.session_state.external_factors_enabled:
    st.subheader("🌍 การตั้งค่าปัจจัยภายนอก (External Factors)")
    
    # ตรวจสอบปัจจัยที่มีในข้อมูล
    available_factors = []
    external_cols = ['temperature', 'humidity', 'holiday_flag', 'campaign', 'outbreak_index', 
                    'population_density', 'school_closed', 'tourists']
    
    for col in external_cols:
        if col in df.columns and not df[col].isna().all():
            available_factors.append(col)
    
    if available_factors:
        # เลือกปัจจัยที่จะใช้
        selected_factors = st.multiselect(
            "เลือกปัจจัยภายนอกที่ต้องการใช้ในการพยากรณ์:",
            available_factors,
            default=available_factors,
            help="ปัจจัยที่เลือกจะถูกรวมเข้าในโมเดล Prophet"
        )
        
        if selected_factors:
            st.success(f"✅ จะใช้ปัจจัยภายนอก: {', '.join(selected_factors)}")
            
            # แสดงสถิติของปัจจัยภายนอก
            st.write("**📊 สถิติปัจจัยภายนอก:**")
            factor_stats = df[selected_factors].describe().round(2)
            st.dataframe(factor_stats, use_container_width=True)
            
            # การตั้งค่าสำหรับการพยากรณ์อนาคต
            st.write("**🔮 การตั้งค่าปัจจัยภายนอกสำหรับการพยากรณ์:**")
            
            future_factors = {}
            
            for factor in selected_factors:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.write(f"**{factor}:**")
                
                with col2:
                    method = st.selectbox(
                        f"วิธีกำหนดค่า {factor}",
                        ["ใช้ค่าเฉลี่ย", "ใช้ค่าล่าสุด", "กำหนดเอง"],
                        key=f"method_{factor}"
                    )
                
                with col3:
                    if method == "ใช้ค่าเฉลี่ย":
                        value = df[factor].mean()
                        st.write(f"ค่าเฉลี่ย: {value:.2f}")
                    elif method == "ใช้ค่าล่าสุด":
                        value = df[factor].iloc[-1]
                        st.write(f"ค่าล่าสุด: {value:.2f}")
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
        'holiday_flag': {'prior_scale': 1.0, 'mode': 'additive'},  # เปลี่ยนจาก holidays
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
            # ใช้ค่า default สำหรับปัจจัยที่ไม่ได้กำหนดไว้
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
with st.spinner("🔄 กำลังเทรนโมเดล Prophet..."):
    model, val_mae, val_mape, has_validation = train_prophet_model_with_factors(prophet_df, selected_factors)

# --- ส่วนสำหรับผู้ใช้ป้อนข้อมูลและพยากรณ์ ---
st.header("🔮 พยากรณ์จำนวนผู้ป่วย")

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
st.subheader("📋 ผลการพยากรณ์")

# แสดงข้อมูล validation ถ้ามี
if has_validation:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Validation MAE", f"{val_mae:.2f}")
    with col2:
        st.metric("Validation MAPE", f"{val_mape:.1f}%")
    with col3:
        if selected_factors:
            st.metric("External Factors", f"{len(selected_factors)} ตัว")
        else:
            st.metric("โมเดล", "พื้นฐาน")

# สร้างตารางผลการพยากรณ์
forecast_display_data = {
    'สัปดาห์ที่': forecast_future['week_num'].astype(int),
    'วันที่': forecast_future['ds'].dt.strftime('%d/%m/%Y'),
    'Prophet พยากรณ์ (ราย)': forecast_future['yhat_adjusted'].round(0).astype(int),
    'Baseline เฉลี่ย (ราย)': [int(recent_avg)] * weeks_to_forecast,
    'ต่างจาก Baseline': (forecast_future['yhat_adjusted'] - recent_avg).round(0).astype(int),
    'ช่วงต่ำ (95% CI)': forecast_future['yhat_lower_adjusted'].round(0).astype(int),
    'ช่วงสูง (95% CI)': forecast_future['yhat_upper_adjusted'].round(0).astype(int)
}

# เพิ่มคอลัมน์ปัจจัยภายนอกถ้ามี
if selected_factors:
    for factor in selected_factors:
        if factor in future_factors:
            forecast_display_data[f'{factor}'] = [future_factors[factor]] * weeks_to_forecast

forecast_display = pd.DataFrame(forecast_display_data)
st.dataframe(forecast_display, use_container_width=True)

# เตือนหากค่าพยากรณ์แตกต่างจาก baseline มากเกินไป
max_diff_percent = abs((forecast_future['yhat_adjusted'] - recent_avg) / recent_avg * 100).max()
if max_diff_percent > 50:
    st.warning(f"⚠️ การพยากรณ์แตกต่างจาก baseline มากถึง {max_diff_percent:.1f}% - ควรตรวจสอบความสมเหตุสมผล")
elif max_diff_percent < 5:
    st.info(f"ℹ️ การพยากรณ์ใกล้เคียง baseline ({max_diff_percent:.1f}%) - โมเดลอาจไม่ได้เพิ่มคุณค่ามากนัก")

# --- แสดงกราฟแนวโน้มและการพยากรณ์แบบเชื่อมต่อ ---
st.subheader("📈 กราฟแนวโน้มและการพยากรณ์")

# สร้างกราฟที่เชื่อมต่อกัน
fig = go.Figure()

# 1. เพิ่มข้อมูลจริง
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=df['cases'],
    mode='lines+markers',
    name='ข้อมูลจริง',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
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
    name='Prophet พยากรณ์',
    line=dict(color='red', width=2),
    marker=dict(size=8, symbol='diamond'),
    hovertemplate='สัปดาห์ที่: %{x}<br>พยากรณ์: %{y:.0f} ราย<extra></extra>'
))

# 3. เพิ่ม Confidence Interval แบบเชื่อมต่อ
ci_upper_connected = [last_cases] + list(forecast_future['yhat_upper_adjusted'])
ci_lower_connected = [last_cases] + list(forecast_future['yhat_lower_adjusted'])

fig.add_trace(go.Scatter(
    x=forecast_weeks_connected + forecast_weeks_connected[::-1],
    y=ci_upper_connected + ci_lower_connected[::-1],
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='ช่วงความเชื่อมั่น 95%',
    showlegend=True,
    hoverinfo='skip'
))

# 4. เพิ่ม Baseline แบบเชื่อมต่อ
baseline_connected = [last_cases] + baseline_forecast
fig.add_trace(go.Scatter(
    x=forecast_weeks_connected,
    y=baseline_connected,
    mode='lines+markers',
    name='Baseline (เฉลี่ย 4 สัปดาห์)',
    line=dict(color='orange', width=2, dash='dot'),
    marker=dict(size=6, symbol='square'),
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
    name='แนวโน้ม (Prophet)',
    line=dict(color='green', dash='dash', width=1),
    opacity=0.7,
    hovertemplate='สัปดาห์ที่: %{x}<br>แนวโน้ม: %{y:.0f} ราย<extra></extra>'
))

# 6. เพิ่มเส้นแบ่งระหว่างข้อมูลจริงกับการพยากรณ์
fig.add_vline(
    x=last_week + 0.5, 
    line_dash="solid", 
    line_color="gray",
    line_width=2,
    annotation_text="จุดเริ่มพยากรณ์",
    annotation_position="top"
)

# ตั้งค่ากราฟ
title = 'แนวโน้มผู้ป่วยและการพยากรณ์ (Facebook Prophet'
if selected_factors:
    title += f' + {len(selected_factors)} External Factors'
title += ')'

fig.update_layout(
    title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='สัปดาห์ที่',
    yaxis_title='จำนวนผู้ป่วย (ราย)',
    hovermode='x unified',
    showlegend=True,
    height=600,
    font=dict(family="kanit, sans-serif", size=12),
    plot_bgcolor='white'
)

# ตั้งค่าช่วงแกน
x_min = max(1, df['week_num'].min() - 1)
x_max = df['week_num'].max() + weeks_to_forecast + 1
fig.update_xaxes(
    range=[x_min, x_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    dtick=max(1, (x_max - x_min) // 20)
)

y_min = 0
y_max = max(df['cases'].max(), forecast_future['yhat_upper'].max()) * 1.1
fig.update_yaxes(
    range=[y_min, y_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray'
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
    st.subheader("📊 ค่าทางสถิติและการประเมินโมเดล")

    # แสดงค่าความแม่นยำของโมเดล
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="MAE",
            value=f"{mae:.2f}",
            help="Mean Absolute Error - ค่าเฉลี่ยของความผิดพลาด"
        )

    with col2:
        st.metric(
            label="RMSE", 
            value=f"{rmse:.2f}",
            help="Root Mean Square Error - รากที่สองของค่าเฉลี่ยความผิดพลาดยกกำลังสอง"
        )

    with col3:
        st.metric(
            label="MAPE",
            value=f"{mape:.1f}%",
            help="Mean Absolute Percentage Error - เปอร์เซ็นต์ความผิดพลาด"
        )

    with col4:
        st.metric(
            label="R²",
            value=f"{r2:.3f}",
            help="R-squared - ค่าสัมประสิทธิ์การตัดสินใจ (0-1, ยิ่งใกล้ 1 ยิ่งดี)"
        )

    # สรุปผลการประเมิน
    if mape < 10:
        accuracy_level = "ดีมาก (MAPE < 10%)"
        accuracy_color = "green"
    elif mape < 20:
        accuracy_level = "ดี (MAPE 10-20%)"
        accuracy_color = "orange" 
    else:
        accuracy_level = "ต้องปรับปรุง (MAPE > 20%)"
        accuracy_color = "red"

    st.info(f"**สรุปความแม่นยำของโมเดล**: {accuracy_level}")

    # แสดงกราฟ Residuals Analysis
    st.subheader("🔍 การวิเคราะห์ Residuals")

    residuals = actual_values - predicted_values

    fig_residuals = go.Figure()

    # กราฟ residuals vs predicted
    fig_residuals.add_trace(go.Scatter(
        x=predicted_values,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='purple', size=8)
    ))

    # เส้น y=0
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    fig_residuals.update_layout(
        title="Residuals vs Predicted Values",
        xaxis_title="ค่าพยากรณ์",
        yaxis_title="Residuals (จริง - พยากรณ์)",
        height=400
    )

    st.plotly_chart(fig_residuals, use_container_width=True)

# --- แสดงสถิติข้อมูลพื้นฐาน ---
st.subheader("📈 สถิติข้อมูลและการพยากรณ์")

col1, col2 = st.columns(2)

with col1:
    st.write("**สถิติข้อมูลในอดีต:**")
    stats_df = pd.DataFrame({
        'สถิติ': ['ค่าเฉลี่ย', 'ค่ามัธยฐาน', 'ส่วนเบียงเบนมาตรฐาน', 'ค่าต่ำสุด', 'ค่าสูงสุด'],
        'ค่า': [
            f"{df['cases'].mean():.1f} ราย",
            f"{df['cases'].median():.1f} ราย", 
            f"{df['cases'].std():.1f} ราย",
            f"{df['cases'].min():.0f} ราย",
            f"{df['cases'].max():.0f} ราย"
        ]
    })
    st.dataframe(stats_df, hide_index=True)

with col2:
    st.write("**สถิติการพยากรณ์:**")
    forecast_stats_df = pd.DataFrame({
        'สถิติ': ['ค่าเฉลี่ย', 'ค่ามัธยฐาน', 'ส่วนเบียงเบนมาตรฐาน', 'ค่าต่ำสุด', 'ค่าสูงสุด'],
        'ค่า': [
            f"{forecast_future['yhat_adjusted'].mean():.1f} ราย",
            f"{forecast_future['yhat_adjusted'].median():.1f} ราย",
            f"{forecast_future['yhat_adjusted'].std():.1f} ราย", 
            f"{forecast_future['yhat_adjusted'].min():.0f} ราย",
            f"{forecast_future['yhat_adjusted'].max():.0f} ราย"
        ]
    })
    st.dataframe(forecast_stats_df, hide_index=True)

# --- แสดงการวิเคราะห์แนวโน้ม ---
st.subheader("📊 การวิเคราะห์แนวโน้ม")

col1, col2, col3 = st.columns(3)

with col1:
    avg_forecast = forecast_future['yhat_adjusted'].mean()
    avg_historical = df['cases'].mean()
    trend_change = ((avg_forecast - avg_historical) / avg_historical) * 100
    
    st.metric(
        label="การเปลี่ยนแปลงค่าเฉลี่ย",
        value=f"{trend_change:+.1f}%",
        delta=f"{avg_forecast - avg_historical:+.1f} ราย"
    )

with col2:
    first_forecast = forecast_future['yhat_adjusted'].iloc[0] 
    last_forecast = forecast_future['yhat_adjusted'].iloc[-1]
    forecast_trend = last_forecast - first_forecast
    
    st.metric(
        label="แนวโน้มในช่วงพยากรณ์",
        value="เพิ่มขึ้น" if forecast_trend > 0 else "ลดลง" if forecast_trend < 0 else "คงที่",
        delta=f"{forecast_trend:+.1f} ราย"
    )

with col3:
    uncertainty = forecast_future['yhat_upper_adjusted'].mean() - forecast_future['yhat_lower_adjusted'].mean()
    st.metric(
        label="ช่วงความไม่แน่นอนเฉลี่ย",
        value=f"±{uncertainty/2:.1f} ราย",
        help="ช่วงความเชื่อมั่น 95% เฉลี่ย"
    )

# --- แสดงกราฟ components ของ Prophet ---
st.subheader("🔧 การวิเคราะห์องค์ประกอบ (Trend & Seasonality)")

try:
    # สร้างกราฟ trend
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
except Exception as e:
    st.warning(f"ไม่สามารถแสดงกราฟ components ได้: {str(e)}")

# แสดงข้อมูลปัจจัยภายนอกที่มีผล
if selected_factors and show_metrics:
    st.subheader("🌍 ผลกระทบของปัจจัยภายนอก")
    
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
            {'ปัจจัย': factor, 'ความสัมพันธ์': f"{corr:.3f}", 'ผลกระทบ': 'สูง' if corr > 0.3 else 'ปานกลาง' if corr > 0.1 else 'ต่ำ'}
            for factor, corr in factor_importance.items()
        ])
        
        st.dataframe(importance_df, use_container_width=True)

st.caption("หมายเหตุ: การพยากรณ์นี้ใช้โมเดล Facebook Prophet ซึ่งสามารถจับ pattern และ seasonality ได้ดีกว่าโมเดลเชิงเส้น")
if selected_factors:
    st.caption(f"โมเดลนี้รวมปัจจัยภายนอก {len(selected_factors)} ตัว เพื่อเพิ่มความแม่นยำ")

# --- Sidebar Information ---
st.sidebar.subheader("ข้อมูลโมเดล")
st.sidebar.info("""
**Facebook Prophet Features:**
- ตรวจจับ seasonality อัตโนมัติ
- รองรับ holiday effects
- มี confidence intervals
- ทนทานต่อ missing data
- ตรวจจับการเปลี่ยนแปลง trend
- รองรับ external regressors
""")

st.sidebar.subheader("🌍 ปัจจัยภายนอกที่รองรับ")
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

st.sidebar.subheader("⚠️ Prophet Reserved Names")
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

st.sidebar.subheader("🔍 ความน่าเชื่อถือของโมเดล")
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

st.sidebar.subheader("📊 ข้อมูลสถิติ")
st.sidebar.info("""
**Metrics:**
- **MAE**: ความผิดพลาดเฉลี่ย
- **RMSE**: รากที่สองของความผิดพลาดกำลังสอง
- **MAPE**: เปอร์เซ็นต์ความผิดพลาด  
- **R²**: ค่าสัมประสิทธิ์การตัดสินใจ

**เกณฑ์ประเมิน MAPE:**
- < 10%: ดีมาก
- 10-20%: ดี
- > 20%: ต้องปรับปรุง

**External Factors:**
- ปรับปรุงความแม่นยำได้ 30-50%
- ต้องมีข้อมูลที่เชื่อถือได้
- ควรมีเหตุผลเชิงสาเหตุ
""")

# แสดงข้อมูลแหล่งข้อมูลปัจจุบัน
st.sidebar.subheader("📊 แหล่งข้อมูลปัจจุบัน")
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
