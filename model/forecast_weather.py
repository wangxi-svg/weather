import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# 1. 读取数据
path = './csv_output/weatherdata7_data.csv'
df = pd.read_csv(path)

# 2. 特征工程
df['观测时间'] = pd.to_datetime(df['观测时间'])
df['月'] = df['观测时间'].dt.month
df['日'] = df['观测时间'].dt.day

label_cols = ['白天天气状况', '晚间天气状况', '白天风力', '夜间风力']
encoders = {}
for col in label_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# 构造滞后特征（前一天温度）
df['前一天最高温度'] = df['最高温度'].shift(1)
df['前一天最低温度'] = df['最低温度'].shift(1)
df = df.dropna()

# 3. 特征与标签
features = ['月', '日', '白天天气状况', '晚间天气状况', '白天风力', '夜间风力',
            '紫外线', '湿度', '能见度', '云量', '前一天最高温度', '前一天最低温度']
X = df[features]
y_high = df['最高温度']
y_low = df['最低温度']

# 4. 拆分数据
X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.2, random_state=42)
_, _, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.2, random_state=42)

# 5. 模型训练
high_model = RandomForestRegressor(random_state=42)
low_model = RandomForestRegressor(random_state=42)
high_model.fit(X_train, y_high_train)
low_model.fit(X_train, y_low_train)

# 6. 模型评估
high_pred = high_model.predict(X_test)
low_pred = low_model.predict(X_test)
high_rmse = sqrt(mean_squared_error(y_high_test, high_pred))
low_rmse = sqrt(mean_squared_error(y_low_test, low_pred))
print(f"最高温度 RMSE: {high_rmse:.2f}")
print(f"最低温度 RMSE: {low_rmse:.2f}")

# 7. 保存模型
os.makedirs('./out', exist_ok=True)
joblib.dump(high_model, './out/high_temp_model.pkl')
joblib.dump(low_model, './out/low_temp_model.pkl')
joblib.dump(encoders, './out/label_encoders.pkl')

# 8. 预测未来3-7天（以最近一天为基础，模拟构造数据）
last_day = df.iloc[-1].copy()
future_days = []
for i in range(3, 8):
    new_day = last_day.copy()
    new_day['观测时间'] = new_day['观测时间'] + pd.Timedelta(days=i)
    new_day['月'] = new_day['观测时间'].month
    new_day['日'] = new_day['观测时间'].day
    # 假设天气状况等维持不变
    new_day['前一天最高温度'] = last_day['最高温度']
    new_day['前一天最低温度'] = last_day['最低温度']
    future_days.append(new_day)

future_df = pd.DataFrame(future_days)
X_future = future_df[features]
future_high = high_model.predict(X_future)
future_low = low_model.predict(X_future)

# 9. 输出预测结果
for i, row in future_df.iterrows():
    print(f"{row['观测时间'].date()} -> 预测最高温度: {future_high[i]:.1f}°C, 最低温度: {future_low[i]:.1f}°C")