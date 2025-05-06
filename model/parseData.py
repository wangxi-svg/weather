import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 创建输出目录
output_dir = './new'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成历史天气数据
def generate_historical_weather():
    # 基础数据
    cities = ['北京', '上海', '广州', '深圳', '成都']
    weather_types = ['晴', '多云', '阴', '小雨', '中雨', '大雨']
    wind_directions = ['东风', '南风', '西风', '北风', '东北风', '西北风', '东南风', '西南风']
    
    # 生成日期序列（从2000年1月1日到2025年5月1日）
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 5, 1)
    days = (end_date - start_date).days + 1
    dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days)]
    
    data = []
    for city in cities:
        for date in dates:
            # 解析日期以调整温度范围
            current_date = datetime.strptime(date, '%Y-%m-%d')
            month = current_date.month
            
            # 根据季节调整温度范围
            if month in [12, 1, 2]:  # 冬季
                max_temp = np.random.uniform(-5, 10)
            elif month in [3, 4, 5]:  # 春季
                max_temp = np.random.uniform(10, 25)
            elif month in [6, 7, 8]:  # 夏季
                max_temp = np.random.uniform(25, 38)
            else:  # 秋季
                max_temp = np.random.uniform(15, 28)
            
            min_temp = max_temp - np.random.uniform(5, 10)
            weather = np.random.choice(weather_types)
            wind = f"{np.random.choice(wind_directions)} {np.random.randint(1, 6)}级"
            
            data.append([len(data)+1, city, date, max_temp, min_temp, weather, wind])
    
    df = pd.DataFrame(data, columns=['id', '城市', '日期', '最高温度', '最低温度', '天气', '风向'])
    df.to_csv(f'{output_dir}/lishiweathers_data.csv', index=False)

# 生成实时天气数据
def generate_current_weather():
    cities = ['北京', '海淀', '朝阳', '西城', '东城']
    weather_conditions = ['晴', '多云', '阴', '小雨']
    
    data = []
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M+08:00')
    
    for city in cities:
        temp = np.random.uniform(15, 25)
        feel_temp = temp - np.random.uniform(2, 4)
        weather = np.random.choice(weather_conditions)
        wind_level = np.random.randint(1, 5)
        humidity = np.random.uniform(30, 70)
        visibility = np.random.uniform(8, 15)
        
        data.append([len(data)+1, city, current_time, temp, feel_temp, 
                    weather, wind_level, humidity, visibility])
    
    df = pd.DataFrame(data, columns=['id', '城市', '时间', '温度', '体感温度', 
                                    '天气情况', '风力等级', '湿度', '能见度'])
    df.to_csv(f'{output_dir}/weatherdata_data.csv', index=False)

# 生成七天天气预报数据
def generate_forecast_weather():
    cities = ['北京', '上海', '广州', '深圳']
    day_conditions = ['晴', '多云', '阴', '小雨']
    night_conditions = ['晴', '多云', '阴']
    
    data = []
    start_date = datetime.now()
    
    for city in cities:
        for day in range(7):
            date = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')
            max_temp = np.random.uniform(20, 30)
            min_temp = max_temp - np.random.uniform(5, 10)
            day_weather = np.random.choice(day_conditions)
            night_weather = np.random.choice(night_conditions)
            wind_level = f"{np.random.randint(1, 3)}-{np.random.randint(3, 5)}"
            rain = np.random.uniform(0, 5) if '雨' in day_weather else 0.0
            uv = np.random.randint(1, 5)
            humidity = np.random.uniform(40, 80)
            visibility = np.random.uniform(10, 25)
            cloud = np.random.uniform(0, 100)
            
            data.append([len(data)+1, city, date, max_temp, min_temp,
                        day_weather, night_weather, wind_level, wind_level,
                        rain, uv, humidity, visibility, cloud])
    
    df = pd.DataFrame(data, columns=['id', '城市', '观测时间', '最高温度', '最低温度',
                                    '白天天气状况', '晚间天气状况', '白天风力', '夜间风力',
                                    '降水量', '紫外线', '湿度', '能见度', '云量'])
    df.to_csv(f'{output_dir}/weatherdata7_data.csv', index=False)

def main():
    print("开始生成模拟数据...")
    
    print("1. 生成历史天气数据...")
    generate_historical_weather()
    
    print("2. 生成实时天气数据...")
    generate_current_weather()
    
    print("3. 生成七天天气预报数据...")
    generate_forecast_weather()
    
    print("数据生成完成！文件保存在 ./new/ 目录下")

if __name__ == "__main__":
    main()