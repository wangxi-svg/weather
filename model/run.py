import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import multiprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

# 读取数据文件
def load_data():
    history_df = pd.read_csv('./new/lishiweathers_data.csv')
    return history_df

# 数据预处理
def preprocess_data(history_df):
    # 处理历史数据
    history_df['日期'] = pd.to_datetime(history_df['日期'])
    history_df['月份'] = history_df['日期'].dt.month
    history_df['季节'] = pd.cut(history_df['月份'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['春', '夏', '秋', '冬'])
    
    # 处理风向数据 - 修复正则表达式
    history_df['风力等级'] = history_df['风向'].str.extract(r'(\d+)').astype(float)
    
    # 添加更多特征
    history_df['温差'] = history_df['最高温度'] - history_df['最低温度']
    history_df['平均温度'] = (history_df['最高温度'] + history_df['最低温度']) / 2
    
    # 标签编码
    le = LabelEncoder()
    history_df['天气_encoded'] = le.fit_transform(history_df['天气'])
    
    # 扩展特征选择
    features = ['月份', '最高温度', '最低温度', '风力等级', '温差', '平均温度']
    X = history_df[features]
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    y = history_df['天气_encoded']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le, features

def train_model(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # XGBoost
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    # LightGBM（可选）
    # model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.1, n_jobs=-1, random_state=42)
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\n模型评估结果:')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'决定系数 (R2): {r2:.4f}')
    # XGBoost特征重要性
    feature_importance = pd.DataFrame({
        '特征': features,
        '重要性': model.feature_importances_
    })
    print('\n特征重要性:')
    print(feature_importance.sort_values('重要性', ascending=False))
    return model

def save_model(model, scaler, le, output_dir='./out'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save model and preprocessors
    joblib.dump(model, os.path.join(output_dir, 'weather_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))
    print(f'\nModel and preprocessors have been saved to: {output_dir}')

def main():
    print("开始加载数据...")
    history_df = load_data()
    print("数据预处理中...")
    X_scaled, y, scaler, le, features = preprocess_data(history_df)
    print("开始训练模型...")
    model = train_model(X_scaled, y, features)
    
    # 保存模型
    save_model(model, scaler, le)
    
    print("\n模型训练和保存完成！")
    
    # 示例预测
    sample_input = np.array([[3, 10, 0, 2]])  # 示例：3月份，最高温10度，最低温0度，风力2级
    prediction = predict_weather(model, scaler, le, sample_input)
    print(f"\n预测天气结果: {prediction[0]}")

if __name__ == "__main__":
    main()





# 训练”天气预报“模型
# 数据文件：
# ./csv/lishiweathers_data.csv
# 内容：
# id,城市,日期,最高温度,最低温度,天气,风向
# 1,北京,2022-01-01,6.0,7.0,晴,东北风 1级
# 2,北京,2022-01-02,2.0,7.0,多云,南风 1级
# ...

# ./csv/weatherdata_data.csv
# 内容：
# id,城市,时间,温度,体感温度,天气情况,风力等级,湿度,能见度
# 1,北京,2024-02-27T16:48+08:00,5.0,2.0,阴,2,37.0,11.0
# 2,海淀,2024-02-27T16:48+08:00,4.0,1.0,阴,1,46.0,10.0
# ...

# ./csv/weatherdata7_data.csv
# 内容：
# id,城市,观测时间,最高温度,最低温度,白天天气状况,晚间天气状况,白天风力,夜间风力,降水量,紫外线,湿度,能见度,云量
# 1,北京,2024-02-24,5.0,-4.0,多云,晴,1-3,1-3,0.0,2.0,45.0,25.0,5.0
# 2,北京,2024-02-25,8.0,-5.0,晴,多云,1-3,1-3,0.0,4.0,56.0,25.0,19.0
# ...

# 添加缺失的预测函数
def predict_weather(model, scaler, le, input_data):
    # 确保输入数据包含所有特征
    if input_data.shape[1] == 4:  # 如果是原始4个特征
        # 计算额外特征
        温差 = input_data[0][1] - input_data[0][2]  # 最高温度 - 最低温度
        平均温度 = (input_data[0][1] + input_data[0][2]) / 2
        # 添加新特征
        input_data = np.column_stack((input_data, 温差, 平均温度))
    
    # 标准化输入数据
    input_scaled = scaler.transform(input_data)
    
    # 预测
    prediction = model.predict(input_scaled)
    
    # 转换预测结果
    weather_prediction = le.inverse_transform(prediction.astype(int))
    return weather_prediction
