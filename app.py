import streamlit as st
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
    page_title="เว็บพยากรณ์โรคไข้หวัดใหญ่",
    page_icon="😷",
    layout="wide"
)

st.title("😷 เว็บพยากรณ์โรคไข้หวัดใหญ่เบื้องต้น")
st.write("เครื่องมือนี้ช่วยพยากรณ์จำนวนผู้ป่วยไข้หวัดใหญ่ในสัปดาห์ข้างหน้า โดยใช้ Facebook Prophet")

# --- 2. การเชื่อมต่อข้อมูลหลายรูปแบบ ---
st.subheader("📊 เชื่อมต่อข้อมูล")

# ใช้ session state เพื่อเก็บข้อมูล
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "ตัวอย่าง"

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
        2. ใส่ข้อมูลตามรูปแบบ:
           ```
           A1: end_date    B1: cases    C1: week_num
           A2: 10/01/2021  B2: 125      C2: 1
           A3: 17/01/2021  B3: 134      C3: 2
           ... และต่อไป
           ```
        
        **ขั้นตอนที่ 2: แชร์ Google Sheets**
        1. คลิกปุ่ม "Share" มุมขวาบน
        2. เปลี่ยน "Restricted" เป็น **"Anyone with the link"**
        3. ตั้งสิทธิ์เป็น **"Viewer"** หรือ **"Editor"**
        4. คลิก "Copy link"
        
        **ขั้นตอนที่ 3: ใส่ URL ด้านล่าง**
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
                            # ทำความสะอาดข้อมูล
                            df_sheets['end_date'] = pd.to_datetime(df_sheets['end_date'], format='%d/%m/%Y', errors='coerce')
                            df_sheets['cases'] = pd.to_numeric(df_sheets['cases'], errors='coerce')
                            df_sheets['week_num'] = pd.to_numeric(df_sheets['week_num'], errors='coerce')
                            
                            # ลบแถวที่มีข้อมูลไม่ครบ
                            df_sheets = df_sheets.dropna().reset_index(drop=True)
                            
                            if len(df_sheets) > 0:
                                # เรียงข้อมูลตามวันที่
                                df_sheets = df_sheets.sort_values('end_date').reset_index(drop=True)
                                
                                # เก็บข้อมูลใน session state
                                st.session_state.current_data = df_sheets
                                st.session_state.data_source = "Google Sheets"
                                
                                st.success(f"✅ เชื่อมต่อ Google Sheets สำเร็จ! {len(df_sheets)} สัปดาห์")
                                
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
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ CSV",
        type=['csv'],
        help="ไฟล์ต้องมีคอลัมน์: end_date, cases, week_num"
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
                # แปลงประเภทข้อมูล
                df_uploaded['end_date'] = pd.to_datetime(df_uploaded['end_date'], format='%d/%m/%Y', errors='coerce')
                df_uploaded['cases'] = pd.to_numeric(df_uploaded['cases'], errors='coerce')
                df_uploaded['week_num'] = pd.to_numeric(df_uploaded['week_num'], errors='coerce')
                
                # ลบแถวที่มีข้อมูลไม่ครบ
                df_uploaded = df_uploaded.dropna().reset_index(drop=True)
                
                if len(df_uploaded) > 0:
                    # เรียงข้อมูลตามวันที่
                    df_uploaded = df_uploaded.sort_values('end_date').reset_index(drop=True)
                    
                    # เก็บข้อมูลใน session state
                    st.session_state.current_data = df_uploaded
                    st.session_state.data_source = f"ไฟล์: {uploaded_file.name}"
                    
                    st.success(f"✅ อัปโหลดไฟล์สำเร็จ! {len(df_uploaded)} สัปดาห์")
                    
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
    st.info("💡 ข้อมูลจำลองสำหรับทดสอบระบบ (52 สัปดาห์)")
    
    # สร้างข้อมูลตัวอย่าง
    dates = pd.date_range(start='2024-01-07', end='2024-12-29', freq='W')
    cases = []
    
    for i, date in enumerate(dates):
        week_of_year = date.isocalendar()[1]
        seasonal = 80 + 30 * np.sin(2 * np.pi * week_of_year / 52)
        noise = np.random.normal(0, 10)
        cases.append(max(10, int(seasonal + noise)))
    
    df_sample = pd.DataFrame({
        'end_date': dates,
        'cases': cases,
        'week_num': range(1, len(dates) + 1)
    })
    
    # เก็บข้อมูลใน session state
    st.session_state.current_data = df_sample
    st.session_state.data_source = "ข้อมูลตัวอย่าง"

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
        st.rerun()
else:
    st.error("❌ ไม่พบข้อมูล กรุณาเลือกแหล่งข้อมูลด้านบน")
    st.stop()

# ตรวจสอบคุณภาพข้อมูล
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

# เตือนถ้าข้อมูลน้อย
if len(df) < 8:
    st.warning("⚠️ ข้อมูลมีน้อยกว่า 8 สัปดาห์ - การพยากรณ์อาจไม่แม่นยำ")

# เตือนถ้ามีค่าผิดปกติ
if df['cases'].max() > df['cases'].mean() * 3:
    st.warning("⚠️ พบค่าผิดปกติในข้อมูล - อาจส่งผลต่อการพยากรณ์")

# --- 3. เตรียมข้อมูลสำหรับ Prophet ---
# Prophet ต้องการคอลัมน์ 'ds' (วันที่) และ 'y' (ค่าที่ต้องการพยากรณ์)
prophet_df = pd.DataFrame({
    'ds': df['end_date'],
    'y': df['cases']
})

# เพิ่ม week_num สำหรับการแสดงผล
prophet_df['week_num'] = df['week_num']

# --- 4. สร้างและเทรนโมเดล Prophet (พร้อม validation) ---
@st.cache_data
def train_and_validate_prophet_model(data):
    # แบ่งข้อมูลเป็น train/test (80/20)
    split_point = int(len(data) * 0.8)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    # สร้างโมเดล Prophet แบบ conservative
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(data) >= 52 else False,  # ปรับตามความยาวข้อมูล
        seasonality_mode='additive',
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # เทรนด้วยข้อมูล train
    model.fit(train_data)
    
    # ทดสอบกับข้อมูล test
    if len(test_data) > 0:
        future_test = model.make_future_dataframe(periods=len(test_data), freq='W')
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
        model_final.fit(data)
        
        return model_final, validation_mae, validation_mape, True
    else:
        return model, None, None, False

model, val_mae, val_mape, has_validation = train_and_validate_prophet_model(prophet_df)

# --- 5. ส่วนสำหรับผู้ใช้ป้อนข้อมูลและพยากรณ์ ---
st.header("🔮 พยากรณ์จำนวนผู้ป่วย")

# จำกัดจำนวนสัปดาห์การพยากรณ์ให้สมเหตุสมผล
max_forecast_weeks = min(12, len(df) // 2)  # ไม่เกิน 12 สัปดาห์ หรือ 50% ของข้อมูลเดิม

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
forecast = model.predict(future)

# แยกข้อมูลการพยากรณ์ (เฉพาะส่วนอนาคต)
forecast_future = forecast.tail(weeks_to_forecast)

# เพิ่ม week_num สำหรับการแสดงผล
last_week_num = df['week_num'].max()
forecast_future = forecast_future.copy()
forecast_future['week_num'] = range(last_week_num + 1, last_week_num + weeks_to_forecast + 1)

# เพิ่มการเปรียบเทียบกับ Simple Baseline (ค่าเฉลี่ย 4 สัปดาห์ล่าสุด)
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

# --- 6. แสดงผลลัพธ์การพยากรณ์ ---
st.subheader("📋 ผลการพยากรณ์")

# แสดงข้อมูล validation ถ้ามี
if has_validation:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Validation MAE", f"{val_mae:.2f}")
    with col2:
        st.metric("Validation MAPE", f"{val_mape:.1f}%")

forecast_display = pd.DataFrame({
    'สัปดาห์ที่': forecast_future['week_num'].astype(int),
    'วันที่': forecast_future['ds'].dt.strftime('%d/%m/%Y'),
    'Prophet พยากรณ์ (ราย)': forecast_future['yhat_adjusted'].round(0).astype(int),
    'Baseline เฉลี่ย (ราย)': [int(recent_avg)] * weeks_to_forecast,
    'ต่างจาก Baseline': (forecast_future['yhat_adjusted'] - recent_avg).round(0).astype(int),
    'ช่วงต่ำ (95% CI)': forecast_future['yhat_lower_adjusted'].round(0).astype(int),
    'ช่วงสูง (95% CI)': forecast_future['yhat_upper_adjusted'].round(0).astype(int)
})
st.dataframe(forecast_display, use_container_width=True)

# เตือนหากค่าพยากรณ์แตกต่างจาก baseline มากเกินไป
max_diff_percent = abs((forecast_future['yhat_adjusted'] - recent_avg) / recent_avg * 100).max()
if max_diff_percent > 50:
    st.warning(f"⚠️ การพยากรณ์แตกต่างจาก baseline มากถึง {max_diff_percent:.1f}% - ควรตรวจสอบความสมเหตุสมผล")
elif max_diff_percent < 5:
    st.info(f"ℹ️ การพยากรณ์ใกล้เคียง baseline ({max_diff_percent:.1f}%) - โมเดลอาจไม่ได้เพิ่มคุณค่ามากนัก")

# --- 7. แสดงกราฟแนวโน้มและการพยากรณ์ด้วย Plotly ---
st.subheader("📈 กราฟแนวโน้มและการพยากรณ์")

# ใช้ week_num แทน date สำหรับ x-axis เพื่อให้เข้าใจง่าย
fig = go.Figure()

# เพิ่มข้อมูลจริง (ใช้ week_num)
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=df['cases'],
    mode='lines+markers',
    name='ข้อมูลจริง',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
    hovertemplate='สัปดาห์ที่: %{x}<br>ผู้ป่วย: %{y} ราย<extra></extra>'
))

# เพิ่มการพยากรณ์ (ใช้ week_num ที่ต่อเนื่อง)
forecast_weeks = range(df['week_num'].max() + 1, df['week_num'].max() + weeks_to_forecast + 1)
fig.add_trace(go.Scatter(
    x=list(forecast_weeks),
    y=forecast_future['yhat_adjusted'],
    mode='lines+markers',
    name='Prophet พยากรณ์',
    line=dict(color='red', width=2),
    marker=dict(size=8, symbol='diamond'),
    hovertemplate='สัปดาห์ที่: %{x}<br>พยากรณ์: %{y:.0f} ราย<extra></extra>'
))

# เพิ่ม Confidence Interval
fig.add_trace(go.Scatter(
    x=list(forecast_weeks) + list(forecast_weeks)[::-1],
    y=list(forecast_future['yhat_upper_adjusted']) + list(forecast_future['yhat_lower_adjusted'][::-1]),
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='ช่วงความเชื่อมั่น 95%',
    showlegend=True,
    hoverinfo='skip'
))

# เพิ่มการพยากรณ์ baseline ในกราฟ
fig.add_trace(go.Scatter(
    x=list(forecast_weeks),
    y=baseline_forecast,
    mode='lines+markers',
    name='Baseline (เฉลี่ย 4 สัปดาห์)',
    line=dict(color='orange', width=2, dash='dot'),
    marker=dict(size=6, symbol='square'),
    hovertemplate='สัปดาห์ที่: %{x}<br>Baseline: %{y:.0f} ราย<extra></extra>'
))

# เพิ่มเส้นแนวโน้มในอดีต
historical_trend = forecast[:len(df)]['yhat']
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=historical_trend,
    mode='lines',
    name='แนวโน้ม (Prophet)',
    line=dict(color='green', dash='dash', width=1),
    opacity=0.7,
    hovertemplate='สัปดาห์ที่: %{x}<br>แนวโน้ม: %{y:.0f} ราย<extra></extra>'
))

# ตั้งค่ากราฟ
fig.update_layout(
    title={
        'text': 'แนวโน้มผู้ป่วยไข้หวัดใหญ่และการพยากรณ์ (Facebook Prophet)',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='สัปดาห์ที่',
    yaxis_title='จำนวนผู้ป่วย (ราย)',
    hovermode='x unified',
    showlegend=True,
    height=600,
    font=dict(family="Arial, sans-serif", size=12),
    plot_bgcolor='white'
)

# ตั้งค่าช่วงแกน X ให้สมเหตุสมผล
x_min = max(1, df['week_num'].min() - 1)
x_max = df['week_num'].max() + weeks_to_forecast + 1
fig.update_xaxes(
    range=[x_min, x_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    dtick=1  # แสดงทุกสัปดาห์
)

# ตั้งค่าช่วงแกน Y ให้สมเหตุสมผล
y_min = 0
y_max = max(df['cases'].max(), forecast_future['yhat_upper'].max()) * 1.1
fig.update_yaxes(
    range=[y_min, y_max],
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray'
)

# แสดงกราฟใน Streamlit
st.plotly_chart(fig, use_container_width=True)

# --- 8. คำนวณค่าทางสถิติของโมเดล ---
try:
    # ใช้ข้อมูลในอดีตเพื่อประเมินความแม่นยำ
    historical_forecast = forecast[forecast['ds'].isin(df['end_date'])]
    
    # ตรวจสอบว่ามีข้อมูลครบหรือไม่
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

# --- 9. แสดงข้อมูลทางสถิติ ---
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

# --- 10. แสดงสถิติข้อมูลพื้นฐาน ---
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

# แสดงการวิเคราะห์แนวโน้ม
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

# แสดงกราฟ components ของ Prophet
st.subheader("🔧 การวิเคราะห์องค์ประกอบ (Trend & Seasonality)")

try:
    # สร้างกราฟ trend
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
except Exception as e:
    st.warning(f"ไม่สามารถแสดงกราฟ components ได้: {str(e)}")

st.caption("หมายเหตุ: การพยากรณ์นี้ใช้โมเดล Facebook Prophet ซึ่งสามารถจับ pattern และ seasonality ได้ดีกว่าโมเดลเชิงเส้น")

# --- Sidebar Information ---
st.sidebar.subheader("ข้อมูลโมเดล")
st.sidebar.info("""
**Facebook Prophet Features:**
- ตรวจจับ seasonality อัตโนมัติ
- รองรับ holiday effects
- มี confidence intervals
- ทนทานต่อ missing data
- ตรวจจับการเปลี่ยนแปลง trend
""")

st.sidebar.subheader("🔍 ความน่าเชื่อถือของโมเดล")
st.sidebar.warning("""
**ข้อจำกัดสำคัญ:**

1. **ข้อมูลจำกัด**: ควรมีข้อมูลอย่างน้อย 1 ปี

2. **ไม่มี External Factors**: ไม่รวมปัจจัยภายนอก เช่น:
   - การระบาดของโรค
   - นโยบายสาธารณสุข
   - การเปลี่ยนแปลงสภาพอากาศ

3. **Extrapolation Risk**: การคาดการณ์ไกลจากข้อมูลเดิม

**คำแนะนำ:**
- ใช้ร่วมกับความรู้ของผู้เชี่ยวชาญ
- ตรวจสอบความสมเหตุสมผล
- อัปเดตโมเดลเมื่อมีข้อมูลใหม่
""")

st.sidebar.subheader("📊 ข้อมูลสถิติ")
st.sidebar.info("""
**Metrics:**
- **MAE**: ความผิดพลาดเฉลี่ย
- **RMSE**: รากที่สองของความผิดพลาดกำลังสอง
- **MAPE**: เปอร์เซ็นต์ความผิดพลาด  
- **R²**: ค่าสัมประสิทธิ์การตัดสินใจ

**Baseline Comparison:**
- เปรียบเทียบกับค่าเฉลี่ย 4 สัปดาห์ล่าสุด
- ช่วยประเมินว่าโมเดลมีประโยชน์มากกว่า simple average หรือไม่

**เกณฑ์ประเมิน MAPE:**
- < 10%: ดีมาก
- 10-20%: ดี
- > 20%: ต้องปรับปรุง
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
