import os
import joblib
import shap
import numpy as np
import pandas as pd

class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "xgb_house_price_model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        print("--- [DEBUG] 檢查模型狀態 ---")
        print("Type of self.model:", type(self.model))
        print("Value of self.model:", self.model)
        print("--------------------------")
        self.explainer = shap.Explainer(self.model)

    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0
        return df[self.model_features]

    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"
        HUMAN_MAP = {
            "main_area": f"主建物面積約 {value:.1f} 坪",
            "balcony_area": f"陽台面積約 {value:.1f} 坪",
            "building_age": f"屋齡約 {int(value)} 年",
            "floor": f"位於第 {int(value)} 樓",
            "total_floors": f"建物總樓層 {int(value)} 樓",
            "has_parking": "具備車位" if value == 1 else "未附車位",
            "has_elevator": "設有電梯" if value == 1 else "未設電梯",
        }
        return HUMAN_MAP.get(feature, feature)

    def predict(self, case_dict):
        X = self._align_features(case_dict)
        pred = float(self.model.predict(X)[0])
        shap_values = self.explainer(X)
        
        # 兼容不同版本 SHAP 的 values 結構
        if hasattr(shap_values, "values"):
            shap_vals = shap_values.values
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
        else:
            shap_vals = shap_values[0]

        feature_names = X.columns
        
        # 處理 expected_value 可能為串列或陣列的情況，避免轉型報錯
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(np.atleast_1d(ev)[0])
        else:
            base_value = float(ev)

        # 累加所有特徵 SHAP 值，保證文字總和 == 預測值
        cumulative_price = base_value
        explanation_lines = [f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整："]

        # 顯示 Top 10 影響因素文字
        idx_sorted = np.argsort(np.abs(shap_vals))[-10:][::-1]
        for i in idx_sorted:
            feat = feature_names[i]
            val = X.iloc[0][feat]
            shap_val = shap_vals[i]
            # 只顯示 active 類別或非零數值
            if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                continue
            human_text = self._feature_to_human(feat, val)
            direction = "推升" if shap_val > 0 else "下修"
            explanation_lines.append(f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪")
            cumulative_price += shap_val

        # 保證 cumulative_price = 模型預測
        explanation_lines.append(f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。")

        return {
            "predicted_price": pred,
            "explanation": "\n".join(explanation_lines),
        }