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
    layout="centered"
)

st.title("😷 เว็บพยากรณ์โรคไข้หวัดใหญ่เบื้องต้น")
st.write("เครื่องมือนี้ช่วยพยากรณ์จำนวนผู้ป่วยไข้หวัดใหญ่ในสัปดาห์ข้างหน้า โดยใช้ Facebook Prophet")

# --- 2. โหลดข้อมูล ---
try:
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/18zRQXwQA9avuIXWaZ7p_jd9dDbTSe-soTMLpmH3_8w4/export?format=csv')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%d/%m/%Y')
    st.subheader("ข้อมูลผู้ป่วยย้อนหลัง")
    st.dataframe(df)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    st.stop()

# --- 3. เตรียมข้อมูลสำหรับ Prophet ---
# Prophet ต้องการคอลัมน์ 'ds' (วันที่) และ 'y' (ค่าที่ต้องการพยากรณ์)
prophet_df = pd.DataFrame({
    'ds': df['end_date'],
    'y': df['cases']
})

# --- 4. สร้างและเทรนโมเดล Prophet ---
@st.cache_data
def train_prophet_model(data):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95  # 95% confidence interval
    )
    model.fit(data)
    return model

model = train_prophet_model(prophet_df)

# --- 5. ส่วนสำหรับผู้ใช้ป้อนข้อมูลและพยากรณ์ ---
st.header("พยากรณ์จำนวนผู้ป่วย")

weeks_to_forecast = st.slider(
    "เลือกจำนวนสัปดาห์ที่ต้องการพยากรณ์ไปข้างหน้า:",
    min_value=1,
    max_value=12,
    value=4
)

# สร้างช่วงวันที่สำหรับการพยากรณ์
future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')
forecast = model.predict(future)

# แยกข้อมูลการพยากรณ์ (เฉพาะส่วนอนาคต)
forecast_future = forecast.tail(weeks_to_forecast)

# --- 6. แสดงผลลัพธ์การพยากรณ์ ---
st.subheader("ผลการพยากรณ์")

forecast_display = pd.DataFrame({
    'วันที่': forecast_future['ds'].dt.strftime('%d/%m/%Y'),
    'จำนวนผู้ป่วยที่พยากรณ์ (ราย)': forecast_future['yhat'].astype(int),
    'ค่าต่ำสุด (95% CI)': forecast_future['yhat_lower'].astype(int),
    'ค่าสูงสุด (95% CI)': forecast_future['yhat_upper'].astype(int)
})
st.dataframe(forecast_display)

# --- 7. แสดงกราฟแนวโน้มและการพยากรณ์ด้วย Plotly ---
st.subheader("กราฟแนวโน้มและการพยากรณ์")

fig = go.Figure()

# เพิ่มข้อมูลจริง
fig.add_trace(go.Scatter(
    x=df['end_date'],
    y=df['cases'],
    mode='lines+markers',
    name='ข้อมูลจริง',
    line=dict(color='blue', width=2),
    marker=dict(size=6)
))

# เพิ่มการพยากรณ์
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines+markers',
    name='พยากรณ์',
    line=dict(color='red', width=2),
    marker=dict(size=8, symbol='diamond')
))

# เพิ่ม Confidence Interval
fig.add_trace(go.Scatter(
    x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
    y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='ช่วงความเชื่อมั่น 95%',
    showlegend=True
))

# เพิ่มเส้นแนวโน้มทั้งหมด (รวมอดีตและอนาคต)
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='แนวโน้มรวม',
    line=dict(color='green', dash='dash', width=1),
    opacity=0.7
))

# ตั้งค่ากราฟ
fig.update_layout(
    title={
        'text': 'แนวโน้มผู้ป่วยไข้หวัดใหญ่และการพยากรณ์ (Facebook Prophet)',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='วันที่',
    yaxis_title='จำนวนผู้ป่วย (ราย)',
    hovermode='x unified',
    showlegend=True,
    height=600,
    font=dict(family="Arial, sans-serif", size=12),
    plot_bgcolor='white'
)

# เพิ่ม grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# แสดงกราฟใน Streamlit
st.plotly_chart(fig, use_container_width=True)

# --- 8. คำนวณค่าทางสถิติของโมเดล ---
# ใช้ข้อมูลในอดีตเพื่อประเมินความแม่นยำ
historical_forecast = forecast.iloc[:len(df)]
actual_values = df['cases'].values
predicted_values = historical_forecast['yhat'].values

# คำนวณค่า error metrics
mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
r2 = r2_score(actual_values, predicted_values)

# --- 9. แสดงข้อมูลทางสถิติ ---
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

# แสดงข้อมูลสถิติพื้นฐาน
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
            f"{forecast_future['yhat'].mean():.1f} ราย",
            f"{forecast_future['yhat'].median():.1f} ราย",
            f"{forecast_future['yhat'].std():.1f} ราย", 
            f"{forecast_future['yhat'].min():.0f} ราย",
            f"{forecast_future['yhat'].max():.0f} ราย"
        ]
    })
    st.dataframe(forecast_stats_df, hide_index=True)

# แสดงการวิเคราะห์แนวโน้ม
st.subheader("📊 การวิเคราะห์แนวโน้ม")

col1, col2, col3 = st.columns(3)

with col1:
    avg_forecast = forecast_future['yhat'].mean()
    avg_historical = df['cases'].mean()
    trend_change = ((avg_forecast - avg_historical) / avg_historical) * 100
    
    st.metric(
        label="การเปลี่ยนแปลงค่าเฉลี่ย",
        value=f"{trend_change:+.1f}%",
        delta=f"{avg_forecast - avg_historical:+.1f} ราย"
    )

with col2:
    first_forecast = forecast_future['yhat'].iloc[0] 
    last_forecast = forecast_future['yhat'].iloc[-1]
    forecast_trend = last_forecast - first_forecast
    
    st.metric(
        label="แนวโน้มในช่วงพยากรณ์",
        value="เพิ่มขึ้น" if forecast_trend > 0 else "ลดลง" if forecast_trend < 0 else "คงที่",
        delta=f"{forecast_trend:+.1f} ราย"
    )

with col3:
    uncertainty = forecast_future['yhat_upper'].mean() - forecast_future['yhat_lower'].mean()
    st.metric(
        label="ช่วงความไม่แน่นอนเฉลี่ย",
        value=f"±{uncertainty/2:.1f} ราย",
        help="ช่วงความเชื่อมั่น 95% เฉลี่ย"
    )

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

# --- 10. แสดงข้อมูลเพิ่มเติมเกี่ยวกับโมเดล ---
st.subheader("ข้อมูลเพิ่มเติม")

# แสดงกราฟ components ของ Prophet
st.subheader("การวิเคราะห์องค์ประกอบ (Trend & Seasonality)")

# สร้างกราฟ trend
fig_components = model.plot_components(forecast)
st.pyplot(fig_components)

st.caption("หมายเหตุ: การพยากรณ์นี้ใช้โมเดล Facebook Prophet ซึ่งสามารถจับ pattern และ seasonality ได้ดีกว่าโมเดลเชิงเส้น")

# --- (Optional) ส่วนสำหรับอัปโหลดไฟล์ ---
st.sidebar.subheader("อัปโหลดข้อมูลใหม่ (ตัวเลือกเสริม)")

uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV ที่มีคอลัมน์ 'end_date' และ 'cases'", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'end_date' in uploaded_df.columns and 'cases' in uploaded_df.columns:
        st.sidebar.write("อัปโหลดข้อมูลสำเร็จ! โปรดรีเฟรชหน้าเว็บเพื่อใช้ข้อมูลใหม่")
    else:
        st.sidebar.error("ไฟล์ CSV ต้องมีคอลัมน์ 'end_date' และ 'cases'")

# --- Model Performance Info ---
st.sidebar.subheader("ข้อมูลโมเดล")
st.sidebar.info("""
**Facebook Prophet Features:**
- ตรวจจับ seasonality อัตโนมัติ
- รองรับ holiday effects
- มี confidence intervals
- ทนทานต่อ missing data
- ตรวจจับการเปลี่ยนแปลง trend
""")

st.sidebar.subheader("ค่าทางสถิติที่แสดง")
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
""")
