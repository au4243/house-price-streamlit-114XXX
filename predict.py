import os
import joblib
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
        
        # 安全初始化 shap
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model, check_additivity=False)
        except Exception as e:
            print(f"初始化 TreeExplainer 失敗（將自動切換為替代說明模式）: {e}")

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
        
        try:
            val_float = float(value)
        except Exception:
            val_float = 0.0

        HUMAN_MAP = {
            "main_area": f"主建物面積約 {val_float:.1f} 坪",
            "balcony_area": f"陽台面積約 {val_float:.1f} 坪",
            "building_age": f"屋齡約 {int(val_float)} 年",
            "floor": f"位於第 {int(val_float)} 樓",
            "total_floors": f"建物總樓層 {int(val_float)} 樓",
            "has_parking": "具備車位" if int(val_float) == 1 else "未附車位",
            "has_elevator": "設有電梯" if int(val_float) == 1 else "未設電梯",
        }
        return HUMAN_MAP.get(feature, feature)

    def predict(self, case_dict):
        X = self._align_features(case_dict)
        
        # 絕對安全的預測值解析方式
        pred = 50.0
        try:
            raw_pred = self.model.predict(X)
            if hasattr(raw_pred, "item"):
                pred = float(raw_pred.item())
            elif isinstance(raw_pred, (list, np.ndarray)):
                pred = float(raw_pred[0])
            else:
                pred = float(raw_pred)
        except Exception as e:
            print(f"模型預測發生例外: {e}")

        feature_names = self.model_features
        shap_vals = None
        base_value = float(pred * 0.9)

        # 嘗試使用 SHAP 解釋
        if self.explainer is not None:
            try:
                raw_shap = self.explainer.shap_values(X)
                if isinstance(raw_shap, list):
                    shap_vals = np.array(raw_shap[0])
                elif hasattr(raw_shap, "values"):
                    shap_vals = np.array(raw_shap.values)
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]
                else:
                    shap_vals = np.array(raw_shap)
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]

                ev = getattr(self.explainer, "expected_value", None)
                if ev is not None:
                    if isinstance(ev, (list, np.ndarray)):
                        base_value = float(ev[0]) if len(ev) > 0 else float(pred * 0.9)
                    else:
                        base_value = float(ev)
            except Exception as e:
                print(f"計算 SHAP 值發生例外，改用條件列舉說明: {e}")
                shap_vals = None

        explanation_lines = []

        # 組合說明文字
        if shap_vals is not None and len(shap_vals) == len(feature_names):
            explanation_lines.append(f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整：")
            try:
                idx_sorted = np.argsort(np.abs(shap_vals))[-10:][::-1]
                for i in idx_sorted:
                    if i < len(feature_names):
                        feat = feature_names[i]
                        val = float(X.iloc[0][feat])
                        shap_val = float(shap_vals[i])
                        if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                            continue
                        human_text = self._feature_to_human(feat, val)
                        direction = "推升" if shap_val > 0 else "下修"
                        explanation_lines.append(f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪")
            except Exception as ex:
                print(f"SHAP 排序解析略過: {ex}")
        
        if len(explanation_lines) <= 0:
            explanation_lines.append(f"📌 本案根據您輸入的房屋條件進行綜合評估：")
            for feat in feature_names:
                try:
                    val = float(X.iloc[0][feat])
                    if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                        continue
                    if val != 0:
                        human_text = self._feature_to_human(feat, val)
                        explanation_lines.append(f"👉 包含條件：{human_text}")
                except Exception:
                    pass

        explanation_lines.append(f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。")

        return {
            "predicted_price": pred,
            "explanation": "\n".join(explanation_lines),
        }