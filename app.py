import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="VANET Congestion Prediction",
    page_icon="🚀",
    layout="wide"
)

# 2. Tải mô hình và bộ chuẩn hóa
@st.cache_resource
def load_assets():
    model = joblib.load('vanet_nbc_model.pkl')
    scaler = joblib.load('vanet_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. Tiêu đề và Mô tả song ngữ
st.title("🚀 VANET Network Performance & Congestion Analysis")
st.subheader("Hệ thống Phân tích Hiệu năng Truyền thông và Dự báo Trạng thái Mạng VANET")

st.markdown("""
*This system utilizes **Naive Bayes (NBC)** and **Logistic Regression** models to analyze 24 network and traffic features to predict congestion levels in real-time.*
*(Hệ thống sử dụng mô hình **Naive Bayes** và **Logistic Regression** để phân tích 24 đặc trưng mạng và giao thông nhằm dự báo mức độ tắc nghẽn theo thời gian thực.)*
""")
st.info(" 💻 Ngành: Điện tử Viễn thông - Nghiên cứu tối ưu hóa mạng thông tin di động")

st.markdown("---")

# 4. Danh sách 24 đặc trưng song ngữ
features_map = {
    'avg_speed_kmph': 'Tốc độ TB (km/h) - Avg Speed',
    'density_veh_per_km': 'Mật độ xe (xe/km) - Vehicle Density',
    'avg_wait_time_s': 'Thời gian chờ TB (s) - Avg Wait Time',
    'occupancy_pct': 'Tỉ lệ chiếm dụng (%) - Occupancy',
    'flow_veh_per_hr': 'Lưu lượng xe/giờ - Traffic Flow',
    'queue_length_veh': 'Độ dài hàng đợi - Queue Length',
    'avg_accel_ms2': 'Gia tốc TB (m/s2) - Avg Acceleration',
    'heading_deg': 'Hướng di chuyển - Heading',
    'signal_state_num': 'Trạng thái tín hiệu - Signal State',
    'incident_num': 'Số vụ sự cố - Incident Count',
    'temp_c': 'Nhiệt độ (°C) - Temperature',
    'visibility_km': 'Tầm nhìn (km) - Visibility',
    'rain_intensity_mmph': 'Cường độ mưa (mm/h) - Rain Intensity',
    'channel_busy_ratio_pct': 'Tỉ lệ kênh bận (%) - CBR',
    'msg_rate_hz': 'Tốc độ thông điệp (Hz) - Message Rate',
    'avg_comm_delay_ms': 'Trễ truyền thông (ms) - Comm Delay',
    'rssi_dbm': 'Cường độ tín hiệu (dBm) - RSSI',
    'packet_loss_pct': 'Tỉ lệ mất gói (%) - Packet Loss',
    'speed_density_ratio': 'Tỉ lệ Tốc độ/Mật độ - Speed-Density Ratio',
    'congestion_pressure': 'Áp lực tắc nghẽn - Congestion Pressure',
    'wireless_congestion_intensity': 'Cường độ mạng - Wireless Intensity',
    'throughput_per_queued_vehicle': 'Thông lượng/xe - Throughput/Veh',
    'acceleration_directionality': 'Tính hướng gia tốc - Accel Direction',
    'weather_factor': 'Yếu tố thời tiết - Weather Factor'
}

# --- PHẦN 5: GIAO DIỆN NHẬP LIỆU ---
with st.form("prediction_form"):
    st.markdown("### 📝 Network & Traffic Parameters (Tham số Mạng & Giao thông)")
    col1, col2 = st.columns(2)
    
    input_data = {}
    feature_keys = list(features_map.keys())
    
    # --- TRONG PHẦN 5 ---
    for idx, key in enumerate(feature_keys):
         label = features_map[key]
         with col1 if idx < 12 else col2:
         # Quan trọng: value=None giúp ô nhập liệu trống hoàn toàn
            input_data[key] = st.number_input(label, value=None, placeholder="Chưa xác định...", format="%.4f")
            
    st.markdown("---")
    submit = st.form_submit_button("🔍 ANALYZE & PREDICT (PHÂN TÍCH & DỰ BÁO)")

# --- PHẦN 6: XỬ LÝ DỰ ĐOÁN ---
# Chỉ thực hiện khi người dùng nhấn nút "submit" của Form
if submit:
    try:
        # Tạo một bản sao để xử lý, tránh ảnh hưởng đến input gốc
        final_input = {}
        
        for key in feature_keys:
            # KIỂM TRA: Nếu người dùng để trống (None), lấy giá trị Mean bù vào
            if input_data[key] is None:
                final_input[key] = mean_data[key]
            else:
                final_input[key] = input_data[key]
        # 1. Chuyển dữ liệu về DataFrame
        df_input = pd.DataFrame([input_data])
        
        # 2. Chuẩn hóa dữ liệu
        input_scaled = scaler.transform(df_input)
        
        # 3. Thực hiện dự đoán
        pred = model.predict(input_scaled)[0]
        probs = np.max(model.predict_proba(input_scaled)) * 100
        
        # 4. Định nghĩa nhãn kết quả
        labels = {
            0: "🚨 GRIDLOCK (Nghẽn mạng mức độ cao)",
            1: "⚠️ HEAVY (Kênh truyền bão hòa)",
            2: "✅ MODERATE (Kết nối ổn định)",
            3: "🟢 FREE-FLOW (Hiệu năng truyền dẫn tối ưu)"
        }
        
        # 5. Hiển thị kết quả (Toàn bộ phần này phải nằm TRONG lệnh IF SUBMIT)
        st.markdown("---")
        st.markdown("### 📊 Network State Analysis")
        
        color = "red" if pred <= 1 else "green"
        st.subheader(f"Status: :{color}[{labels[pred]}]")
        st.write(f"**Confidence (Độ tin cậy):** {probs:.2f}%")
        st.progress(probs / 100)

    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra: {e}")
else:
    # Nếu chưa nhấn nút, có thể hiện một thông báo nhẹ nhàng
    st.write("👈 *Please enter parameters and click 'ANALYZE' to see results.*")
    st.write("*(Vui lòng nhập các thông số và nhấn 'ANALYZE' để xem kết quả.)*")
