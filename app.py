import streamlit as st
from predict import HousePricePredictor

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="房價估價系統",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 房價估價系統（文字版）｜依據 114 年 不動產成交資料")
st.caption("XGBoost + 可解釋 AI（SHAP）")

# =========================
# 行政區對照表 (維持不變)
# =========================
CITY_TOWN_MAP = {
    "臺北市": [
        "士林區", "大同區", "大安區", "中山區", "中正區", "內湖區",
        "文山區", "北投區", "松山區", "信義區", "南港區", "萬華區",
    ],
    "新北市": [
        "八里區", "三芝區", "三重區", "三峽區", "土城區", "中和區", "五股區",
        "平溪區", "永和區", "石門區", "石碇區", "汐止區", "坪林區", "板橋區",
        "林口區", "金山區", "泰山區", "烏來區", "貢寮區", "淡水區", "深坑區",
        "新店區", "新莊區", "瑞芳區", "萬里區", "樹林區", "雙溪區", "蘆洲區", "鶯歌區",
    ],
    # ... (其餘縣市對照表保留您原本的內容)
}

# =========================
# 載入模型
# =========================
@st.cache_resource
def load_predictor():
    return HousePricePredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"模型載入失敗，請確認檔案 `xgb_house_price_model.pkl` 與 `model_features.pkl` 是否存在。錯誤訊息：{e}")
    st.stop()

# =========================
# 側邊欄輸入
# =========================
st.sidebar.header("📋 房屋基本資料")

city = st.sidebar.selectbox("縣市", list(CITY_TOWN_MAP.keys()))
town = st.sidebar.selectbox("鄉鎮市區", CITY_TOWN_MAP[city])
district = f"{city}{town}"
st.sidebar.caption(f"📍 行政區：{district}")

building_type = st.sidebar.selectbox(
    "建物型態", ["住宅大樓", "華廈", "公寓", "透天厝"]
)
main_use = st.sidebar.selectbox(
    "主要用途", ["住家用", "商業用", "住商用"]
)
building_age = st.sidebar.number_input("屋齡（年）", 0, 80, 20)
main_area = st.sidebar.number_input("主建物面積（坪）", 5.0, 100.0, 30.0)
balcony_area = st.sidebar.number_input("陽台面積（坪）", 0.0, 20.0, 5.0)
floor = st.sidebar.number_input("所在樓層", 1, 100, 5)
total_floors = st.sidebar.number_input("總樓層數", 1, 100, 10)
has_parking = st.sidebar.radio("是否有車位", ["有", "無"])
has_elevator = st.sidebar.radio("是否有電梯", ["有", "無"])

# =========================
# 組合輸入資料
# =========================
case_dict = {
    "district": district,
    "building_type": building_type,
    "main_use": main_use,
    "building_age": building_age,
    "main_area": main_area,
    "balcony_area": balcony_area,
    "floor": floor,
    "total_floors": total_floors,
    "has_parking": 1 if has_parking == "有" else 0,
    "has_elevator": 1 if has_elevator == "有" else 0,
}

# =========================
# 主畫面
# =========================
st.subheader("📊 預測結果")

if "result" not in st.session_state:
    st.session_state.result = None

if st.button("🚀 開始估價"):
    with st.spinner("模型預測中，請稍候..."):
        st.session_state.result = predictor.predict(case_dict)

# =========================
# 顯示文字結果
# =========================
if st.session_state.result is not None:
    result = st.session_state.result
    st.markdown("## 📝 中文估價說明")
    st.text(result["explanation"])

    st.success(
        f"💰 模型最終預估單價：約 **{result['predicted_price']:.1f} 萬 / 坪**"
    )
else:
    st.info("👈 請先填寫左側資料，並點擊「開始估價」")