#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 通用天气预测模型训练脚本

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np

# 1. 加载数据
data_path = './csv_output/weatherdata7_data.csv'
df = pd.read_csv(data_path)

# 清洗列名（去除可能的空格）
df.columns = df.columns.str.strip()

# 2. 重命名必要的列
df = df.rename(columns={
    '城市': 'City',
    '观测时间': 'Date',
    '最高温度': 'MaxTemp',
    '最低温度': 'MinTemp',
    '白天天气状况': 'Weather',
    '白天风力': 'Wind',
    '湿度': 'Humidity',
    '能见度': 'Visibility',
    '云量': 'Cloud'
})

# 3. 时间转换与特征构造
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday  # 星期几

# 4. 创建滞后特征（前一天最高/最低温度）
df.sort_values(['City', 'Date'], inplace=True)
df['MaxTemp_prev'] = df.groupby('City')['MaxTemp'].shift(1)
df['MinTemp_prev'] = df.groupby('City')['MinTemp'].shift(1)

# 5. 编码分类特征
city_encoder = LabelEncoder()
weather_encoder = LabelEncoder()
wind_encoder = LabelEncoder()

df['City_enc'] = city_encoder.fit_transform(df['City'])
df['Weather_enc'] = weather_encoder.fit_transform(df['Weather'])
df['Wind_enc'] = wind_encoder.fit_transform(df['Wind'].astype(str))

# 6. 删除包含空值的行
df.dropna(inplace=True)

# 7. 构造特征列与标签列
features = [
    'City_enc', 'Month', 'Day', 'Weekday',
    'MaxTemp_prev', 'MinTemp_prev', 'Wind_enc',
    'Humidity', 'Visibility', 'Cloud'
]
X = df[features]
y_max = df['MaxTemp']
y_min = df['MinTemp']
y_weather = df['Weather_enc']

# 8. 训练模型
model_max = RandomForestRegressor(n_estimators=100, random_state=42)
model_max.fit(X, y_max)

model_min = RandomForestRegressor(n_estimators=100, random_state=42)
model_min.fit(X, y_min)

model_weather = RandomForestClassifier(n_estimators=100, random_state=42)
model_weather.fit(X, y_weather)
# 回归模型评估函数
def evaluate_regression(y_true, y_pred, label):
    print(f'\n📊 [{label}] 回归模型评估:')
    print('R²:', round(r2_score(y_true, y_pred), 4))
    print('MAE:', round(mean_absolute_error(y_true, y_pred), 4))
    print('RMSE:', round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))

# 分类模型评估函数
def evaluate_classification(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    print(f'\n📊 [{label}] 分类模型评估:')
    print('Accuracy:', round(acc, 4))

# 预测结果
y_max_pred = model_max.predict(X)
y_min_pred = model_min.predict(X)
y_weather_pred = model_weather.predict(X)

# 输出评估指标
evaluate_regression(y_max, y_max_pred, '最高温度')
evaluate_regression(y_min, y_min_pred, '最低温度')
evaluate_classification(y_weather, y_weather_pred, '白天天气')

# 9. 保存模型和编码器
out_dir = './out'
os.makedirs(out_dir, exist_ok=True)

joblib.dump(model_max, os.path.join(out_dir, 'max_temp_model.pkl'))
joblib.dump(model_min, os.path.join(out_dir, 'min_temp_model.pkl'))
joblib.dump(model_weather, os.path.join(out_dir, 'weather_model.pkl'))

joblib.dump(city_encoder, os.path.join(out_dir, 'city_encoder.pkl'))
joblib.dump(weather_encoder, os.path.join(out_dir, 'weather_encoder.pkl'))
joblib.dump(wind_encoder, os.path.join(out_dir, 'wind_encoder.pkl'))

print('✅ 模型训练完成并保存到目录:', out_dir)

# 📊 [最高温度] 回归模型评估:
# R²: 0.9958
# MAE: 0.5229
# RMSE: 0.7139

# 📊 [最低温度] 回归模型评估:
# R²: 0.9968
# MAE: 0.4315
# RMSE: 0.5931

# 📊 [白天天气] 分类模型评估:
# Accuracy: 1.0