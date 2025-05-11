import pandas as pd
import joblib
from datetime import datetime, timedelta

# 加载模型与编码器
model_max = joblib.load('./out/max_temp_model.pkl')
model_min = joblib.load('./out/min_temp_model.pkl')
model_weather = joblib.load('./out/weather_model.pkl')

city_encoder = joblib.load('./out/city_encoder.pkl')
weather_encoder = joblib.load('./out/weather_encoder.pkl')
wind_encoder = joblib.load('./out/wind_encoder.pkl')

# 预测函数
def predict_weather(city_name, last_date_str, last_max_temp, last_min_temp, wind_level, humidity, visibility, cloud, n_days):
    assert n_days <= 7, "最多只能预测 7 天"

    # 编码输入特征
    city_id = city_encoder.transform([city_name])[0]
    wind_id = wind_encoder.transform([str(wind_level)])[0]

    # 初始数据
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
    max_temp = last_max_temp
    min_temp = last_min_temp

    predictions = []

    for i in range(1, n_days + 1):
        predict_date = last_date + timedelta(days=i)
        month = predict_date.month
        day = predict_date.day
        weekday = predict_date.weekday()

        # 构建输入特征
        X_pred = pd.DataFrame([[
            city_id, month, day, weekday,
            max_temp, min_temp, wind_id,
            humidity, visibility, cloud
        ]], columns=[
            'City_enc', 'Month', 'Day', 'Weekday',
            'MaxTemp_prev', 'MinTemp_prev', 'Wind_enc',
            'Humidity', 'Visibility', 'Cloud'
        ])

        # 预测
        max_temp = model_max.predict(X_pred)[0]
        min_temp = model_min.predict(X_pred)[0]
        weather_class = model_weather.predict(X_pred)[0]
        weather_label = weather_encoder.inverse_transform([weather_class])[0]

        predictions.append({
            '日期': predict_date.strftime('%Y-%m-%d'),
            '城市': city_name,
            '预测最高温度': round(max_temp, 1),
            '预测最低温度': round(min_temp, 1),
            '预测天气': weather_label
        })

    return predictions

# 示例调用
if __name__ == '__main__':
    results = predict_weather(
        city_name='北京',
        last_date_str='2024-02-25',
        last_max_temp=8.0,
        last_min_temp=-5.0,
        wind_level='1-3',
        humidity=56.0,
        visibility=25.0,
        cloud=19.0,
        n_days=5
    )

    for r in results:
        print(r)