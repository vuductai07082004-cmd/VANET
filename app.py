import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="VANET Congestion Prediction",
    page_icon="🚀",
    layout="wide"
)

# 2. Tải mô hình và bộ chuẩn hóa
@st.cache_resource
def load_assets():
    try:
        # Kiểm tra file tồn tại trước khi load để tránh lỗi thầm lặng
        files = ['vanet_nbc_model.pkl', 'vanet_scaler.pkl', 'vanet_means.pkl']
        for f in files:
            if not os.path.exists(f):
                st.error(f"Thiếu file hệ thống: {f}")
                return None, None, None
        
        model = joblib.load('vanet_nbc_model.pkl')
        scaler = joblib.load('vanet_scaler.pkl')
        mean_data = joblib.load('vanet_means.pkl')
        return model, scaler, mean_data
    except Exception as e:
        st.error(f"Lỗi tải file hệ thống: {e}")
        return None, None, None

model, scaler, mean_data = load_assets()

# 3. Tiêu đề
st.title("🚀 VANET Network Performance & Congestion Analysis")
st.subheader("Hệ thống Phân tích Hiệu năng Truyền thông và Dự báo Trạng thái Mạng VANET")

st.markdown("""
*System utilizes **Naive Bayes (NBC)** to predict congestion levels in real-time.*
""")
st.info(" 💻 Ngành: Điện tử Viễn thông - UTC")
st.markdown("---")

# 4. Danh sách 24 đặc trưng
features_map = {
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
    'acceleration_directionality': 'Tính hướng gia tốc', 'weather_factor': 'Yếu tố thời tiết'
}

# 5. Giao diện nhập liệu
with st.form("prediction_form"):
    st.markdown("### 📝 Network & Traffic Parameters")
    col1, col2 = st.columns(2)
    input_data = {}
    feature_keys = list(features_map.keys())
    
    for idx, key in enumerate(feature_keys):
        label = features_map[key]
        with col1 if idx < 12 else col2:
            # Giá trị mặc định lấy từ mean_data nếu có, nếu không thì để 0.0
            default_val = float(mean_data[key]) if mean_data is not None else 0.0
            input_data[key] = st.number_input(label, value=default_val, format="%.4f")
            
    st.markdown("---")
    submit = st.form_submit_button("🔍 ANALYZE & PREDICT")

# 6. Xử lý dự đoán
if submit:
    if model is None or scaler is None:
        st.error("❌ Không thể dự báo vì model hoặc scaler chưa được tải thành công.")
    else:
        try:
            # Chuyển đổi dữ liệu input sang DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Đảm bảo thứ tự các cột đúng như lúc train model
            df_input = df_input[feature_keys]
            
            # Chuẩn hóa
            input_scaled = scaler.transform(df_input)
            
            # Dự đoán
            pred = model.predict(input_scaled)[0]
            probs = np.max(model.predict_proba(input_scaled)) * 100
            
            labels = {
                0: "🚨 NETWORK CONGESTION (Nghẽn mạng mức độ cao)",
                1: "⚠️ CHANNEL SATURATION (Kênh truyền bão hòa)",
                2: "✅ STABLE CONNECTIVITY (Kết nối ổn định)",
                3: "🟢 OPTIMAL PERFORMANCE (Hiệu năng truyền dẫn tối ưu)"
            }
            
            st.markdown("### 📊 Network State Analysis")
            color = "red" if pred <= 1 else "green"
            st.subheader(f"Status: :{color}[{labels.get(pred, 'Unknown')}]")
            st.write(f"**Confidence:** {probs:.2f}%")
            st.progress(probs / 100)

        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {e}")
else:
    st.write("👈 *Please enter parameters and click 'ANALYZE' to see results.*")
