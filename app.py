import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cấu hình trang Web
st.set_page_config(page_title="VANET Congestion Predictor", layout="wide")

# 1. Tải mô hình
@st.cache_resource
def load_models():
    model = joblib.load('vanet_nbc_model.pkl')
    scaler = joblib.load('vanet_scaler.pkl')
    return model, scaler

model, scaler = load_models()

# 2. Danh sách 24 đặc trưng
features_vn = {
    'avg_speed_kmph': 'Tốc độ TB (km/h)', 'density_veh_per_km': 'Mật độ xe (xe/km)',
    'avg_wait_time_s': 'Thời gian chờ TB (s)', 'occupancy_pct': 'Tỉ lệ chiếm dụng (%)',
    'flow_veh_per_hr': 'Lưu lượng xe/giờ', 'queue_length_veh': 'Độ dài hàng đợi',
    'avg_accel_ms2': 'Gia tốc TB (m/s2)', 'heading_deg': 'Hướng di chuyển',
    'signal_state_num': 'Trạng thái tín hiệu', 'incident_num': 'Số vụ sự cố',
    'temp_c': 'Nhiệt độ (°C)', 'visibility_km': 'Tầm nhìn (km)',
    'rain_intensity_mmph': 'Cường độ mưa (mm/h)', 'channel_busy_ratio_pct': 'Tỉ lệ kênh bận (%)',
    'msg_rate_hz': 'Tốc độ thông điệp (Hz)', 'avg_comm_delay_ms': 'Trễ truyền thông (ms)',
    'rssi_dbm': 'Cường độ tín hiệu (dBm)', 'packet_loss_pct': 'Tỉ lệ mất gói (%)',
    'speed_density_ratio': 'Tỉ lệ Tốc độ/Mật độ', 'congestion_pressure': 'Áp lực tắc nghẽn',
    'wireless_congestion_intensity': 'Cường độ mạng', 'throughput_per_queued_vehicle': 'Thông lượng/xe',
    'acceleration_directionality': 'Hướng gia tốc', 'weather_factor': 'Yếu tố thời tiết'
}

st.title("🚀 Hệ thống Dự báo Tắc nghẽn VANET (UTC)")
st.markdown("---")

# 3. Giao diện nhập liệu (Sử dụng Sidebar hoặc Form để không phải load lại từng bước)
st.sidebar.header("Nhập thông số đầu vào")
input_data = {}

# Chia làm 2 cột để giao diện đẹp hơn
col1, col2 = st.columns(2)

for idx, (en, vn) in enumerate(features_vn.items()):
    with col1 if idx < 12 else col2:
        # Streamlit cho phép nhập trực tiếp, không cần lệnh "back" vì bạn có thể click vào bất kỳ ô nào để sửa
        input_data[en] = st.number_input(f"{idx+1}. {vn}", value=0.0, format="%.4f")

# 4. Nút Dự đoán
if st.button("CHẨN ĐOÁN TRẠNG THÁI GIAO THÔNG", type="primary"):
    # Chuyển dữ liệu thành DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Chuẩn hóa và dự đoán
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)[0]
    probs = np.max(model.predict_proba(input_scaled)) * 100
    
    # Kết quả
    traffic_map = {0: "🚨 GRIDLOCK", 1: "⚠️ HEAVY", 2: "✅ MODERATE", 3: "🟢 FREE-FLOW"}
    colors = {0: "red", 1: "orange", 2: "blue", 3: "green"}
    
    st.markdown("---")
    st.subheader("KẾT QUẢ PHÂN TÍCH")
    st.write(f"Trạng thái dự báo: **:{colors[prediction]}[{traffic_map[prediction]}]**")
    st.progress(probs / 100)
    st.write(f"Độ tin cậy của mô hình NBC: **{probs:.2f}%**")