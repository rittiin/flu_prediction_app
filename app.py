import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# --- เพิ่มโค้ดส่วนนี้เพื่อตั้งค่า Font สำหรับภาษาไทย ---
import matplotlib.font_manager as fm

# ตรวจสอบว่ามี Font ภาษาไทยที่คุณต้องการใช้หรือไม่
# (เปลี่ยน 'TH Sarabun New' เป็นชื่อ Font ที่มีในเครื่องของคุณ ถ้าใช้ Windows)
# (สำหรับ Mac อาจจะเป็น 'Thonburi' หรือ 'Ayuthaya')
# (สำหรับ Linux อาจจะเป็น 'Noto Sans Thai' หรือ 'Liberation Sans')
thai_font_name = 'Kanit' # ลองเปลี่ยนชื่อ Font ถ้ายังไม่แสดงผล
# ถ้าไม่แน่ใจว่ามี Font อะไรบ้าง ลองรันโค้ดด้านล่างในเซลล์ Jupyter แยกต่างหาก
# for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
#     try:
#         fprop = fm.FontProperties(fname=font)
#         if fprop.get_name() == thai_font_name:
#             print(f"Found {thai_font_name} at: {font}")
#             break
#     except:
#         pass
# else:
#     st.warning(f"ไม่พบ Font '{thai_font_name}' ในระบบ อาจต้องติดตั้ง Font หรือใช้ Font อื่น")

plt.rcParams['font.family'] = thai_font_name # กำหนด Font หลัก
plt.rcParams['axes.unicode_minus'] = False # แก้ไขเครื่องหมายลบภาษาไทย
# --- สิ้นสุดโค้ดตั้งค่า Font ---

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

# --- 6. แสดงกราฟแนวโน้มและการพยากรณ์ ---
st.subheader("กราฟแนวโน้มและการพยากรณ์")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['week_num'], df['cases'], marker='o', linestyle='-', color='blue', label='ข้อมูลจริง')
ax.plot(X, model.predict(X), color='green', linestyle='--', label='เส้นแนวโน้ม')
ax.plot(forecast_weeks_num, predicted_cases, marker='x', linestyle=':', color='red', label='พยากรณ์')

ax.set_title("แนวโน้มผู้ป่วยไข้หวัดใหญ่และการพยากรณ์")
ax.set_xlabel("สัปดาห์ที่")
ax.set_ylabel("จำนวนผู้ป่วย (ราย)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

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
