import pickle
import pandas as pd
import numpy as np
import shap

class HousePricePredictor:
    def __init__(self, model_path="xgb_house_price_model.pkl", features_path="model_features.pkl"):
        # 載入 XGBoost 模型與模型訓練時的特徵欄位名稱
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(features_path, "rb") as f:
            self.model_features = pickle.load(f)
            
        # 建立 SHAP Explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, case_dict):
        """
        case_dict 包含：
        district, building_type, main_use, building_age, 
        main_area, balcony_area, floor, total_floors, has_parking, has_elevator
        """
        # 1. 建立對應的原始特徵 DataFrame
        # 注意：根據模型特徵，可能包含 building_area_sqm（平方公尺），若是以坪為單位需自行換算或對齊
        main_area_sqm = case_dict["main_area"] * 3.30578  # 坪轉平方公尺（若訓練特徵包含 building_area_sqm）
        balcony_area_sqm = case_dict["balcony_area"] * 3.30578

        raw_data = {
            "building_age": [case_dict["building_age"]],
            "building_area_sqm": [main_area_sqm + balcony_area_sqm], # 視模型實際訓練欄位而定
            "floor": [case_dict["floor"]],
            "total_floors": [case_dict["total_floors"]],
            "main_area": [case_dict["main_area"]],
            "balcony_area": [case_dict["balcony_area"]],
            "has_parking": [case_dict["has_parking"]],
            "has_elevator": [case_dict["has_elevator"]]
        }
        
        df = pd.DataFrame(raw_data)

        # 2. 處理類別型特徵的 One-Hot Encoding (對應 371 個特徵)
        # 建立全為 0 的 DataFrame，欄位為 model_features
        input_encoded = pd.DataFrame(0, index=[0], columns=self.model_features)
        
        # 填入數值特徵
        for col in df.columns:
            if col in input_encoded.columns:
                input_encoded[col] = df[col]

        # 處理行政區 One-Hot
        district_col = f"district_{case_dict['district']}"
        if district_col in input_encoded.columns:
            input_encoded[district_col] = 1

        # 處理建物型態 One-Hot (注意名稱需與訓練集一致)
        # 例如：building_type_住宅大樓(...)
        for col in input_encoded.columns:
            if case_dict["building_type"] in col:
                input_encoded[col] = 1
            if case_dict["main_use"] in col:
                input_encoded[col] = 1

        # 3. 模型預測
        pred_price = self.model.predict(input_encoded)[0]

        # 4. 產出 SHAP 解釋說明文字
        shap_values = self.explainer(input_encoded)
        
        # 簡單組合中文解釋說明
        explanation = f"本案位於 {case_dict['district']}，屋齡 {case_dict['building_age']} 年，主建物面積 {case_dict['main_area']} 坪。"
        explanation += f"\n經 XGBoost 模型綜合評估周邊行情、樓層（{case_dict['floor']}/{case_dict['total_floors']} 樓）與各項設施後得出此預估價。"

        return {
            "predicted_price": float(pred_price),
            "explanation": explanation
        }