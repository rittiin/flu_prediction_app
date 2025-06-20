import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# --- เปลี่ยนจาก Matplotlib เป็น Plotly ---
import plotly.express as px
import plotly.graph_objects as go # อาจจะไม่จำเป็นสำหรับกรณีนี้ แต่มีไว้เผื่อ

# ไม่ต้อง import matplotlib.pyplot และ matplotlib.font_manager แล้ว
# ไม่ต้องมีโค้ด plt.rcParams[...] หรือ fm._rebuild() แล้ว (นี่คือข้อดี!)

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
    df = pd.read_csv('https://docs.google.com/sheets/d/18zRQXwQA9avuIXWaZ7p_jd9dDbTSe-soTMLpmH3_8w4/export?format=csv')
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

# --- 6. แสดงกราฟแนวโน้มและการพยากรณ์ (ใช้ Plotly) ---
st.subheader("กราฟแนวโน้มและการพยากรณ์")

# สร้าง DataFrame สำหรับ Plotly
# รวมข้อมูลจริงและข้อมูลพยากรณ์เข้าด้วยกัน
plot_data = pd.DataFrame({
    'สัปดาห์ที่': df['week_num'].tolist() + [int(w[0]) for w in forecast_weeks_num],
    'จำนวนผู้ป่วย (ราย)': df['cases'].tolist() + [int(c) for c in predicted_cases],
    'ประเภทข้อมูล': ['ข้อมูลจริง'] * len(df) + ['พยากรณ์'] * len(forecast_weeks_num)
})

# เพิ่มข้อมูลเส้นแนวโน้มจากโมเดล
# สร้างชุดข้อมูลสำหรับเส้นแนวโน้มจากข้อมูลจริง
trend_line_data = pd.DataFrame({
    'สัปดาห์ที่': X['week_num'],
    'จำนวนผู้ป่วย (ราย)': model.predict(X),
    'ประเภทข้อมูล': ['เส้นแนวโน้ม'] * len(X)
})

# รวม DataFrame ทั้งหมดเข้าด้วยกัน
combined_df = pd.concat([plot_data, trend_line_data], ignore_index=True)


# สร้างกราฟด้วย Plotly Express
# ใช้ color เพื่อแยกประเภทข้อมูล (ข้อมูลจริง, พยากรณ์, เส้นแนวโน้ม)
# สร้างกราฟด้วย Plotly Express
fig = px.line(
    combined_df,
    x='สัปดาห์ที่',
    y='จำนวนผู้ป่วย (ราย)',
    color='ประเภทข้อมูล',
    title='แนวโน้มผู้ป่วยไข้หวัดใหญ่และการพยากรณ์',
    labels={
        'สัปดาห์ที่': 'สัปดาห์ที่',
        'จำนวนผู้ป่วย (ราย)': 'จำนวนผู้ป่วย (ราย)',
        'ประเภทข้อมูล': 'Legend'
    },
    line_dash='ประเภทข้อมูล',
    markers=True
)

# ... (โค้ด update_traces) ...

# **ส่วนสำคัญ: ปรับแต่ง layout และ Font สำหรับ Plotly**
fig.update_layout(
    # ลองใช้ Noto Sans Thai เป็นตัวแรกและตัวเดียวไปก่อน เพื่อตัดตัวแปร
    font_family="Noto Sans Thai, sans-serif",
    # เพิ่มการกำหนด font สำหรับ title, axis titles, legend titles โดยเฉพาะ
    title_font=dict(family="Noto Sans Thai, sans-serif", size=20),
    xaxis_title_font=dict(family="Noto Sans Thai, sans-serif", size=16),
    yaxis_title_font=dict(family="Noto Sans Thai, sans-serif", size=16),
    legend_title_font=dict(family="Noto Sans Thai, sans-serif", size=12), # เพิ่มสำหรับ legend title
    legend_font=dict(family="Noto Sans Thai, sans-serif", size=10) # เพิ่มสำหรับ legend item text
)


st.plotly_chart(fig, use_container_width=True) # แสดงกราฟ Plotly ใน Streamlit

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
