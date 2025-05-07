import pandas as pd
import joblib
def load_artifacts(model_path='./model/out/weather_clf_model.joblib',
                  scaler_path='./model/out/weather_scaler.joblib',
                  labels_path='./model/out/weather_labels.joblib'):
    """加载训练好的模型、标准化器和标签"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    labels = joblib.load(labels_path)
    return model, scaler, labels

def preprocess_input(raw_data, window_size=3):
    """
    预处理输入数据（需要包含至少3天历史数据）
    :param raw_data: 包含字段 ['日期','天气','最高温度','最低温度','风向'] 的DataFrame
    :return: 处理后的特征数组 (1, n_features)
    """
    df = raw_data.copy()
    
    # 基础特征工程
    df['日期'] = pd.to_datetime(df['日期'])
    df['月份'] = df['日期'].dt.month
    df['季节'] = df['月份'].map({1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4})
    df['风力等级'] = df['风向'].str.extract(r'(\d+)').astype(float)
    df['温差'] = df['最高温度'] - df['最低温度']
    
    # 滚动特征
    df['3日平均温度'] = df['最高温度'].rolling(window_size).mean()
    df['前日天气'] = df['天气'].shift(1).ffill().bfill()

    # 特征列定义（必须与训练时完全一致）
    feature_cols = ['最高温度', '最低温度', '风力等级', '温差',
                   '月份', '季节', '3日平均温度', '前日天气']
    
    # 动态编码天气特征
    encoded = pd.get_dummies(df[feature_cols], columns=['前日天气'])
    
    # 对齐特征维度（补充缺失的天气类别列）
    expected_weather_cols = [f'前日天气_{l}' for l in joblib.load('./model/out/weather_labels.joblib')]
    for col in expected_weather_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    # 确保列顺序一致（数值特征在前 + 天气类别按字母序）
    non_weather = [col for col in encoded.columns if not col.startswith('前日天气_')]
    ordered_cols = non_weather + sorted([col for col in encoded.columns if col.startswith('前日天气_')])
    encoded = encoded[ordered_cols]

    # 提取最近window_size天的数据并展平
    if len(encoded) < window_size:
        raise ValueError(f"需要至少{window_size}天的历史数据")
    return encoded.iloc[-window_size:].values.flatten().reshape(1, -1)

def predict_weather(model, scaler, input_features):
    """执行预测"""
    scaled = scaler.transform(input_features)
    return model.predict(scaled)[0]

def test_prediction():
    """测试预测流程"""
    # 加载模型组件
    model, scaler, labels = load_artifacts()
    
    # 构造测试数据集（5组不同场景的测试用例）
    test_cases = [
        # 冬季测试用例
        pd.DataFrame([
            {'日期': '2023-12-01', '天气': '晴', '最高温度': 5, '最低温度': -2, '风向': '北风3级'},
            {'日期': '2023-12-02', '天气': '多云', '最高温度': 4, '最低温度': -3, '风向': '西北风4级'},
            {'日期': '2023-12-03', '天气': '阴', '最高温度': 3, '最低温度': -4, '风向': '北风2级'}
        ]),
        # 夏季测试用例
        pd.DataFrame([
            {'日期': '2023-07-15', '天气': '晴', '最高温度': 35, '最低温度': 26, '风向': '东南风2级'},
            {'日期': '2023-07-16', '天气': '多云', '最高温度': 33, '最低温度': 25, '风向': '东风3级'},
            {'日期': '2023-07-17', '天气': '阴', '最高温度': 32, '最低温度': 24, '风向': '东南风2级'}
        ]),
        # 春季测试用例
        pd.DataFrame([
            {'日期': '2023-04-10', '天气': '小雨', '最高温度': 18, '最低温度': 12, '风向': '南风4级'},
            {'日期': '2023-04-11', '天气': '阴', '最高温度': 20, '最低温度': 13, '风向': '东南风3级'},
            {'日期': '2023-04-12', '天气': '多云', '最高温度': 22, '最低温度': 14, '风向': '南风2级'}
        ]),
        # 秋季测试用例
        pd.DataFrame([
            {'日期': '2023-10-05', '天气': '多云', '最高温度': 25, '最低温度': 15, '风向': '西北风3级'},
            {'日期': '2023-10-06', '天气': '晴', '最高温度': 23, '最低温度': 14, '风向': '北风4级'},
            {'日期': '2023-10-07', '天气': '阴', '最高温度': 21, '最低温度': 13, '风向': '西北风2级'}
        ]),
        # 雨雪天气测试用例
        pd.DataFrame([
            {'日期': '2023-01-20', '天气': '小雪', '最高温度': 0, '最低温度': -5, '风向': '北风5级'},
            {'日期': '2023-01-21', '天气': '中雪', '最高温度': -2, '最低温度': -7, '风向': '东北风4级'},
            {'日期': '2023-01-22', '天气': '大雪', '最高温度': -3, '最低温度': -8, '风向': '北风6级'}
        ])
    ]
    
    # 对每组测试用例进行预测
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"测试日期: {test_data['日期'].iloc[-1]}")
        try:
            features = preprocess_input(test_data)
            pred = predict_weather(model, scaler, features)
            print(f"✅ 预测成功！预测结果：{pred}")
        except Exception as e:
            print(f"❌ 预测失败：{str(e)}")

if __name__ == '__main__':
    test_prediction()