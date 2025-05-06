# readModel.py
import numpy as np
import joblib
def predict_next_days(model, le, weather_data, forecast_days=3):
    """
    根据传入的天气数据预测未来天气
    weather_data: 包含7天天气数据的列表，每个元素是字典，包含最高温度、最低温度、风向等信息
    """
    if len(weather_data) != 7:
        raise ValueError("必须提供最近7天的天气数据。")
    
    # 打印最近7天数据
    print("\n最近7天的天气数据：")
    for day in weather_data:
        print(f"日期: {day['日期']}, 最高温度: {day['最高温度']}°C, "
              f"最低温度: {day['最低温度']}°C, 天气: {day['天气']}, 风向: {day['风向']}")
    
    # 处理数据
    processed_data = []
    for day in weather_data:
        # 提取风力等级
        wind_level = float(day['风向'].split('级')[0][-1]) if '级' in day['风向'] else 0
        temp_diff = day['最高温度'] - day['最低温度']
        avg_temp = (day['最高温度'] + day['最低温度']) / 2
        processed_data.extend([day['最高温度'], day['最低温度'], wind_level, temp_diff, avg_temp])
    
    # 转换为模型输入格式
    input_data = np.array(processed_data).reshape(1, -1)
    prediction = model.predict(input_data)[0][:forecast_days]
    predicted_weather = le.inverse_transform(np.round(prediction).astype(int))
    return predicted_weather

def main():
    # 载入模型
    model = joblib.load('./out/multi_day_weather_model.joblib')
    le = joblib.load('./out/label_encoder.joblib')

    # 示例天气数据（最近7天）
    example_weather = [
        {'日期': '2024-04-01', '最高温度': 25, '最低温度': 15, '天气': '晴', '风向': '东北风3级'},
        {'日期': '2024-04-02', '最高温度': 26, '最低温度': 16, '天气': '多云', '风向': '东风2级'},
        {'日期': '2024-04-03', '最高温度': 24, '最低温度': 14, '天气': '晴', '风向': '西北风4级'},
        {'日期': '2024-04-04', '最高温度': 23, '最低温度': 13, '天气': '阴', '风向': '南风3级'},
        {'日期': '2024-04-05', '最高温度': 22, '最低温度': 12, '天气': '小雨', '风向': '东南风2级'},
        {'日期': '2024-04-06', '最高温度': 21, '最低温度': 11, '天气': '多云', '风向': '北风3级'},
        {'日期': '2024-04-07', '最高温度': 24, '最低温度': 14, '天气': '晴', '风向': '西风2级'},
    ]

    # 预测3/5/7天
    for days in [3, 5, 7]:
        result = predict_next_days(model, le, example_weather, forecast_days=days)
        print(f"\n预测未来{days}天天气：{list(result)}")

if __name__ == '__main__':
    main()