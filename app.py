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
st.title("🚀 VANET Traffic Congestion Prediction System")
st.subheader("Hệ thống Dự báo Tắc nghẽn Giao thông trong mạng VANET")

st.markdown("""
*This system utilizes **Naive Bayes (NBC)** and **Logistic Regression** models to analyze 24 network and traffic features to predict congestion levels in real-time.*
*(Hệ thống sử dụng mô hình **Naive Bayes** và **Logistic Regression** để phân tích 24 đặc trưng mạng và giao thông nhằm dự báo mức độ tắc nghẽn theo thời gian thực.)*
""")
st.info(" UTC University - Research Project Demo (Đồ án Nghiên cứu)")

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

# 5. Giao diện nhập liệu chia làm 2 cột
with st.form("prediction_form"):
    st.markdown("### 📝 Input Features (Nhập các thông số đặc trưng)")
    col1, col2 = st.columns(2)
    
    input_data = {}
    feature_keys = list(features_map.keys())
    
    for idx, key in enumerate(feature_keys):
        label = features_map[key]
        with col1 if idx < 12 else col2:
            input_data[key] = st.number_input(label, value=0.0, format="%.4f")
            
    st.markdown("---")
    submit = st.form_submit_button("🔍 ANALYZE & PREDICT (PHÂN TÍCH & DỰ BÁO)")

# 6. Xử lý kết quả
if submit:
    # Chuẩn bị dữ liệu
    df_input = pd.DataFrame([input_data])
    
    # Chuẩn hóa
    input_scaled = scaler.transform(df_input)
    
    # Dự đoán
    pred = model.predict(input_scaled)[0]
    probs = np.max(model.predict_proba(input_scaled)) * 100
    
    # Map kết quả song ngữ
    labels = {
        0: "🚨 GRIDLOCK (Kẹt xe nghiêm trọng)",
        1: "⚠️ HEAVY (Giao thông đông đúc)",
        2: "✅ MODERATE (Giao thông ổn định)",
        3: "🟢 FREE-FLOW (Thông thoáng)"
    }
    
    # Hiển thị kết quả
    st.markdown("### 📊 Prediction Result (Kết quả dự báo)")
    
    # Chọn màu sắc cho kết quả
    color = "red" if pred <= 1 else "green"
    
    st.subheader(f"Status: :{color}[{labels[pred]}]")
    st.write(f"**Confidence (Độ tin cậy):** {probs:.2f}%")
    
    # Hiển thị thanh tiến trình độ tin cậy
    st.progress(probs / 100)
    
    st.balloons() # Hiệu ứng chúc mừng khi dự đoán xong
