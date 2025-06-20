import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="เว็บพยากรณ์โรคไข้หวัดใหญ่",
    page_icon="😷",
    layout="centered"
)

st.title("😷 เว็บพยากรณ์โรคไข้หวัดใหญ่เบื้องต้น")
st.write("เครื่องมือนี้ช่วยพยากรณ์จำนวนผู้ป่วยไข้หวัดใหญ่ในสัปดาห์ข้างหน้า โดยใช้ข้อมูลย้อนหลัง")

# --- 2. โหลดข้อมูล (ตัวอย่าง: จาก CSV) ---
try:
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/18zRQXwQA9avuIXWaZ7p_jd9dDbTSe-soTMLpmH3_8w4/export?format=csv')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%d/%m/%Y')
    st.subheader("ข้อมูลผู้ป่วยย้อนหลัง")
    st.dataframe(df)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'flu_data.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในไดเรกทอรีเดียวกันกับ 'app.py'")
    st.stop()

# --- 3. สร้างโมเดลพยากรณ์ (Linear Regression) ---
X = df[['week_num']]
y = df['cases']
model = LinearRegression()
model.fit(X, y)

# --- 4. ส่วนสำหรับผู้ใช้ป้อนข้อมูลและพยากรณ์ ---
st.header("พยากรณ์จำนวนผู้ป่วย")

weeks_to_forecast = st.slider(
    "เลือกจำนวนสัปดาห์ที่ต้องการพยากรณ์ไปข้างหน้า:",
    min_value=1,
    max_value=10,
    value=3
)

last_week_num = df['week_num'].max()
forecast_weeks_num = np.array([last_week_num + i for i in range(1, weeks_to_forecast + 1)]).reshape(-1, 1)
predicted_cases = model.predict(forecast_weeks_num)

# --- 5. แสดงผลลัพธ์การพยากรณ์ ---
st.subheader("ผลการพยากรณ์")

forecast_df = pd.DataFrame({
    'สัปดาห์ที่พยากรณ์': [int(w[0]) for w in forecast_weeks_num],
    'จำนวนผู้ป่วยที่พยากรณ์ (ราย)': [int(c) for c in predicted_cases]
})
st.dataframe(forecast_df)

# --- 6. แสดงกราฟแนวโน้มและการพยากรณ์ด้วย Plotly ---
st.subheader("กราฟแนวโน้มและการพยากรณ์")

# สร้างกราฟด้วย Plotly
fig = go.Figure()

# เพิ่มข้อมูลจริง
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=df['cases'],
    mode='lines+markers',
    name='ข้อมูลจริง',
    line=dict(color='blue'),
    marker=dict(size=8)
))

# เพิ่มเส้นแนวโน้ม
trend_line = model.predict(X)
fig.add_trace(go.Scatter(
    x=df['week_num'],
    y=trend_line,
    mode='lines',
    name='เส้นแนวโน้ม',
    line=dict(color='green', dash='dash')
))

# เพิ่มการพยากรณ์
fig.add_trace(go.Scatter(
    x=[w[0] for w in forecast_weeks_num],
    y=predicted_cases,
    mode='lines+markers',
    name='พยากรณ์',
    line=dict(color='red', dash='dot'),
    marker=dict(size=10, symbol='x')
))

# ตั้งค่ากราฟ
fig.update_layout(
    title={
        'text': 'แนวโน้มผู้ป่วยไข้หวัดใหญ่และการพยากรณ์',
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

# เพิ่ม grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# แสดงกราฟใน Streamlit
st.plotly_chart(fig, use_container_width=True)

st.caption("หมายเหตุ: การพยากรณ์นี้เป็นแบบจำลองเชิงเส้นอย่างง่าย อาจไม่แม่นยำหากมีปัจจัยซับซ้อนอื่น ๆ")

# --- (Optional) ส่วนสำหรับอัปโหลดไฟล์ ---
st.sidebar.subheader("อัปโหลดข้อมูลใหม่ (ตัวเลือกเสริม)")

uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV ที่มีคอลัมน์ 'week_num' และ 'cases'", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'week_num' in uploaded_df.columns and 'cases' in uploaded_df.columns:
        st.sidebar.write("อัปโหลดข้อมูลสำเร็จ! โปรดรีเฟรชหน้าเว็บเพื่อใช้ข้อมูลใหม่")
    else:
        st.sidebar.error("ไฟล์ CSV ต้องมีคอลัมน์ 'week_num' และ 'cases'")
