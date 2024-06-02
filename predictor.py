import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np

class Inference:
    def __init__(self, model_path, sc_path):
        self.model_path = model_path
        self.sc_path = sc_path

        if os.path.exists(self.model_path) and os.path.exists(self.sc_path):
            self.model = pickle.load(open(self.model_path, "rb"))
            self.sc = pickle.load(open(self.sc_path, "rb"))
        else:
            print("Model or Standard Scaler path is not correct!")

    def get_string_to_datetime(self, date):
        dt = datetime.strptime(date, "%d/%m/%Y")
        return {"day": dt.day, "month": dt.month, "year": dt.year, "week_day": dt.strftime("%A")}

    def season_to_df(self, seasons):
        seasons_cols = ['Spring', 'Summer', 'Winter']
        seasons_data = np.zeros((1, len(seasons_cols)), dtype="int")

        df_seasons = pd.DataFrame(seasons_data, columns=seasons_cols)
        if seasons in seasons_cols:
            df_seasons[seasons] = 1
        return df_seasons

    def days_df(self, week_day):
        days_names = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        days_name_data = np.zeros((1, len(days_names)), dtype="int")

        df_days = pd.DataFrame(days_name_data, columns=days_names)

        if week_day in days_names:
            df_days[week_day] = 1
        return df_days

    def prepare_dataframe(self, date, hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, seasons, holiday, functioning_day):
        holiday_dic = {"No Holiday": 0, "Holiday": 1}
        functioning_day_dic = {"No": 0, "Yes": 1}

        str_to_date = self.get_string_to_datetime(date)

        u_input_list = [hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, holiday_dic[holiday], functioning_day_dic[functioning_day], str_to_date["day"], str_to_date["month"], str_to_date["year"]]

        # Ensure feature names are in lowercase to match the training data
        features_name = ["Hour", 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'day', 'month', 'year']

        df_u_input = pd.DataFrame([u_input_list], columns=features_name)
        df_seasons = self.season_to_df(seasons)
        df_days = self.days_df(str_to_date["week_day"])

        df_for_pred = pd.concat([df_u_input, df_seasons, df_days], axis=1)

        return df_for_pred

    def predict(self, df):
        scaled_data = self.sc.transform(df)
        prediction = self.model.predict(scaled_data)
        return prediction
