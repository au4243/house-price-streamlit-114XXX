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

        # 載入檔案
        loaded_data = joblib.load(model_path)
        
        print("--- [DEBUG] 檢查載入的原始物件 ---")
        print("Type of loaded_data:", type(loaded_data))
        print("---------------------------------")

        # 支援：如果存進去的是字典，自動把模型抽出來
        if isinstance(loaded_data, dict):
            print("⚠️ 偵測到模型檔是一個字典，正在尋找模型物件...")
            # 常見可能存放模型的鍵值
            possible_keys = ['model', 'regressor', 'xgb', 'estimator', 'best_estimator']
            found = False
            for k in possible_keys:
                if k in loaded_data:
                    self.model = loaded_data[k]
                    print(f"✅ 已成功從字典的 key ['{k}'] 中取得模型！")
                    found = True
                    break
            if not found:
                # 如果找不到指定 key，嘗試抓取字典的第一個值
                first_val = list(loaded_data.values())[0]
                if hasattr(first_val, 'predict'):
                    self.model = first_val
                    print("✅ 已成功從字典的第一個值中取得模型！")
                else:
                    raise TypeError("❌ 無法從載入的字典中識別出含有 predict 方法的模型物件。")
        else:
            self.model = loaded_data

        # 最終確認模型是否有 predict 方法
        if not hasattr(self.model, 'predict'):
            raise TypeError(f"❌ 載入的物件型態為 {type(self.model)}，它沒有 predict 方法，請確認 xgb_house_price_model.pkl 是否正確！")

        self.model_features = joblib.load(feature_path)

        # 安全初始化 SHAP Explainer
        self.explainer = None
        try:
            self.explainer = shap.Explainer(self.model)
        except Exception as e1:
            print(f"⚠️ shap.Explainer 初始化失敗: {e1}，嘗試改用 TreeExplainer...")
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e2:
                print(f"⚠️ shap.TreeExplainer 也初始化失敗: {e2}。將關閉 SHAP 解釋功能，僅提供預測價格。")

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

        if self.explainer is not None:
            try:
                shap_values = self.explainer(X)
                shap_vals = shap_values.values[0]
                feature_names = X.columns
                
                base_val_raw = self.explainer.expected_value
                base_value = float(base_val_raw[0]) if isinstance(base_val_raw, (list, np.ndarray)) else float(base_val_raw)

                cumulative_price = base_value
                explanation_lines = [f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整："]

                idx_sorted = np.argsort(np.abs(shap_vals))[-10:][::-1]
                for i in idx_sorted:
                    feat = feature_names[i]
                    val = X.iloc[0][feat]
                    shap_val = shap_vals[i]
                    if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                        continue
                    human_text = self._feature_to_human(feat, val)
                    direction = "推升" if shap_val > 0 else "下修"
                    explanation_lines.append(f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪")
                    cumulative_price += shap_val

                explanation_lines.append(f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。")
                return {
                    "predicted_price": pred,
                    "explanation": "\n".join(explanation_lines),
                }
            except Exception as e:
                print(f"⚠️ 計算 SHAP 解釋時發生錯誤: {e}")

        fallback_explanation = (
            f"📌 模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。\n"
            f"（註：目前系統無法載入詳細的特徵歸因解釋，但不影響整體預測結果）"
        )
        return {
            "predicted_price": pred,
            "explanation": fallback_explanation,
        }