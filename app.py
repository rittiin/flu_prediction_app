import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# --- เพิ่มโค้ดส่วนนี้เพื่อตรวจสอบ Font ที่มีอยู่ ---
# (จะแสดงใน Streamlit logs หรือในหน้าเว็บ ถ้าคุณ st.write ออกมา)
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
available_fonts = []
for font_path in font_list:
    try:
        prop = fm.FontProperties(fname=font_path)
        available_fonts.append(prop.get_name())
    except Exception as e:
        # st.warning(f"Error loading font {font_path}: {e}") # อาจจะเยอะไป ถ้ามี error
        pass

# สามารถพิมพ์ลงในหน้า Streamlit เพื่อดูได้ (ชั่วคราว)
# st.subheader("Available Fonts on Server:")
# st.write(sorted(list(set(available_fonts))))

# ตรวจสอบว่ามี Noto Sans Thai หรือไม่
if 'Noto Sans Thai' in available_fonts:
    st.success("พบ 'Noto Sans Thai' ในระบบ!")
    thai_font_name = 'Noto Sans Thai'
else:
    # ลองใช้ Font อื่นที่มักจะมีใน Linux
    if 'Liberation Sans' in available_fonts:
        st.warning("ไม่พบ 'Noto Sans Thai', ใช้ 'Liberation Sans' แทน")
        thai_font_name = 'Liberation Sans'
    elif 'DejaVu Sans' in available_fonts:
        st.warning("ไม่พบ 'Noto Sans Thai', ใช้ 'DejaVu Sans' แทน")
        thai_font_name = 'DejaVu Sans'
    else:
        st.error("ไม่พบ Font ภาษาไทยที่เหมาะสม กรุณาตรวจสอบการติดตั้ง Font")
        thai_font_name = 'sans-serif' # Fallback to generic sans-serif


plt.rcParams['font.family'] = thai_font_name
plt.rcParams['axes.unicode_minus'] = False
# --- สิ้นสุดโค้ดตรวจสอบ Font ---

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="เว็บพยากรณ์โรคไข้หวัดใหญ่",
    page_icon="😷",
    layout="centered"
)

st.title("😷 เว็บพยากรณ์โรคไข้หวัดใหญ่เบื้องต้น")
st.write("เครื่องมือนี้ช่วยพยากรณ์จำนวนผู้ป่วยไข้หวัดใหญ่ในสัปดาห์ข้างหน้า โดยใช้ข้อมูลย้อนหลัง ด้วยโมเดล LinearRegression")

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
