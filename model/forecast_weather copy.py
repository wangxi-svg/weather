import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

def load_and_preprocess(path='./csv/lishiweathers_data.csv'):
    df = pd.read_csv(path)
    df['日期'] = pd.to_datetime(df['日期'])
    df['月份'] = df['日期'].dt.month
    df['季节'] = df['月份'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df['风力等级'] = df['风向'].str.extract(r'(\d+)').astype(float)
    df['温差'] = df['最高温度'] - df['最低温度']
    
    # 增强天气类别合并策略（确保最小样本数）
    weather_mapping = {
        '晴': '晴',
        '多云': '多云',
        '阴': '多云',
        '雨': '雨',
        '雷阵雨': '雨', 
        '阵雨': '雨',
        '小雨': '雨',
        '中雨': '雨',
        '雪': '雪',
        '大雪': '雪',
        '小雪': '雪',
        '雾': '其他',  # 合并稀有天气到其他
        '沙尘': '其他'
    }
    df['天气'] = df['天气'].map(weather_mapping).fillna('其他')
    
    # 确保每个类别至少有2个样本
    weather_counts = df['天气'].value_counts()
    valid_categories = weather_counts[weather_counts >= 2].index
    df = df[df['天气'].isin(valid_categories)]
    
    # 添加滚动统计特征
    df['3日平均温度'] = df['最高温度'].rolling(3).mean()
    df['前日天气'] = df['天气'].shift(1)
    df = df.ffill().bfill()
    
    return df

def create_dataset(df, window_size=3):
    feature_cols = ['最高温度', '最低温度', '风力等级', '温差', 
                   '月份', '季节', '3日平均温度', '前日天气']
    
    # 动态特征编码
    df_encoded = pd.get_dummies(df[feature_cols], columns=['前日天气'])
    
    X, y = [], []
    for i in range(len(df) - window_size - 1):
        hist = df_encoded.iloc[i:i+window_size]
        label = df.iloc[i+window_size]['天气']
        X.append(hist.values.flatten())
        y.append(label)
    return np.array(X), np.array(y)

def train_model(X, y):
    # 调整抽样策略
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15,
            stratify=y,  # 保留分层抽样但已确保类别有效性
            random_state=42
        )
    except ValueError:
        # 回退策略：当分层抽样失败时使用普通抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15,
            random_state=42
        )
    
    # 改进标准化流程
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 优化模型参数
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.2,
        max_depth=5,
        num_leaves=20,
        min_child_samples=5,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )
    
    # 添加样本权重
    class_weights = {k: v for k, v in zip(*np.unique(y_train, return_counts=True))}
    sample_weights = np.array([class_weights[c] for c in y_train])
    
    model.fit(X_train_scaled, y_train, sample_weight=1/sample_weights)
    
    # 输出优化评估
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ 模型准确率: {acc:.4f}")
    print("\n📋 分类报告:\n", classification_report(y_test, y_pred, zero_division=0))
    
    os.makedirs('./out', exist_ok=True)
    joblib.dump(model, './out/weather_clf_model.joblib')
    joblib.dump(scaler, './out/weather_scaler.joblib')
    joblib.dump(np.unique(y).tolist(), './out/weather_labels.joblib')
    return model, scaler

def predict_next_day(model, df, scaler):
    feature_cols = ['最高温度', '最低温度', '风力等级', '温差', 
                   '月份', '季节', '3日平均温度', '前日天气']
    df_encoded = pd.get_dummies(df[feature_cols], columns=['前日天气'])
    
    # 对齐特征维度
    recent = df_encoded[-3:].values.flatten().reshape(1, -1)
    recent_scaled = scaler.transform(recent)
    
    pred = model.predict(recent_scaled)[0]
    print(f"\n📅 预测明天天气：{pred}")
    return pred

def main():
    df = load_and_preprocess()
    X, y = create_dataset(df)
    model, scaler = train_model(X, y)
    predict_next_day(model, df, scaler)

if __name__ == '__main__':
    main()
# ✅ 模型准确率: 0.1279

# 📋 分类报告:
#                precision    recall  f1-score   support

#           其他       0.56      0.05      0.10      7443
#           多云       0.44      0.02      0.03      7491
#            晴       0.29      0.30      0.29      3440
#            雨       0.08      0.76      0.14      1212
#            雪       0.01      0.44      0.03        93

#     accuracy                           0.13     19679
#    macro avg       0.27      0.31      0.12     19679
# weighted avg       0.43      0.13      0.11     19679


# 📅 预测明天天气：晴