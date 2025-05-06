import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# ========== 第一步：读取与预处理 ==========
def load_and_preprocess(path='./new/lishiweathers_data.csv'):
    df = pd.read_csv(path)
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 增加更多时间特征
    df['月份'] = df['日期'].dt.month
    df['星期'] = df['日期'].dt.dayofweek
    df['季节'] = df['月份'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})

    # 风力等级提取
    df['风力等级'] = df['风向'].str.extract(r'(\d+)').astype(float)
    
    # 温差 & 平均温度
    df['温差'] = df['最高温度'] - df['最低温度']
    df['平均温度'] = (df['最高温度'] + df['最低温度']) / 2
    
    # 添加滞后特征
    df['前一天最高温'] = df['最高温度'].shift(1)
    df['前一天最低温'] = df['最低温度'].shift(1)
    df['前一天风力'] = df['风力等级'].shift(1)
    
    # 填充缺失值
    df = df.fillna(method='ffill')

    # 天气编码
    le = LabelEncoder()
    df['天气_encoded'] = le.fit_transform(df['天气'])

    # 保存编码器
    os.makedirs('./out', exist_ok=True)
    joblib.dump(le, './out/label_encoder.joblib')

    return df, le

# ========== 第二步：构建序列样本 ==========
def create_sequences(df, window_size=7, forecast_horizon=3):
    X, y = [], []
    features = ['最高温度', '最低温度', '风力等级', '温差', '平均温度', 
                '月份', '星期', '季节', '前一天最高温', '前一天最低温', '前一天风力']
    
    for i in range(len(df) - window_size - forecast_horizon + 1):
        hist = df.iloc[i:i + window_size]
        future = df.iloc[i + window_size:i + window_size + forecast_horizon]

        hist_features = hist[features].values.flatten()
        future_labels = future['天气_encoded'].values

        X.append(hist_features)
        y.append(future_labels)

    return np.array(X), np.array(y)

# ========== 第三步：训练模型 ==========
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型参数网格
    param_grid = {
        'estimator__n_estimators': [500, 1000],
        'estimator__max_depth': [6, 8],
        'estimator__learning_rate': [0.01, 0.05],
        'estimator__subsample': [0.8, 0.9],
        'estimator__colsample_bytree': [0.8, 0.9],
        'estimator__min_child_weight': [1, 3]
    }
    
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    model = MultiOutputRegressor(base_model)
    
    # 网格搜索找最优参数
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print("最优参数:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # 评估模型
    y_pred = best_model.predict(X_test_scaled)
    y_pred_rounded = np.round(y_pred).astype(int)
    accuracy = np.mean([accuracy_score(y_test[:, i], y_pred_rounded[:, i]) for i in range(y_test.shape[1])])
    print(f"模型准确率: {accuracy:.4f}")

    # 保存模型和标准化器
    joblib.dump(best_model, './out/multi_day_weather_model.joblib')
    joblib.dump(scaler, './out/scaler.joblib')

    return best_model, scaler

# ========== 第四步：进行预测 ==========
def predict_next_days(model, recent_days_df, le, scaler, forecast_days=3):
    if len(recent_days_df) < 7:
        raise ValueError("需要至少7天历史数据进行预测。")

    features = ['最高温度', '最低温度', '风力等级', '温差', '平均温度', 
                '月份', '星期', '季节', '前一天最高温', '前一天最低温', '前一天风力']
    
    # 提取最近7天特征
    input_data = recent_days_df[-7:][features].values.flatten().reshape(1, -1)
    
    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)
    
    # 预测
    prediction = model.predict(input_data_scaled)[0][:forecast_days]
    weather_pred = le.inverse_transform(np.round(prediction).astype(int))

    return weather_pred

# ========== 主函数 ==========
def main():
    print("加载并预处理数据...")
    df, le = load_and_preprocess()

    for forecast_days in [3, 5, 7]:
        print(f"\n=== 使用优化后的 XGBoost 训练并预测未来 {forecast_days} 天 ===")
        X, y = create_sequences(df, window_size=7, forecast_horizon=forecast_days)
        model, scaler = train_model(X, y)

        recent_data = df.copy()
        future_weather = predict_next_days(model, recent_data, le, scaler, forecast_days)
        print(f"预测未来{forecast_days}天天气情况: {future_weather}")

if __name__ == '__main__':
    main()