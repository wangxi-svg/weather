import pymysql
import requests
import json

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "tianqi",
}

# 读取中国城市编码数据
def read_china_city_codes():
    with open("china.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# 爬取7天天气数据
def crawl_7day_weather(city_id):
    base_url = "https://devapi.qweather.com/v7/weather/7d?"
    key = "709f7d5825f2478dbb8de09c5b1171e9"

    params = {
        "location": city_id,
        "key": key,
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if data["code"] == "200":
            return data["daily"][:7]  # 获取连续7天的天气数据
        else:
            print(f"Error: {data['code']}, {data['message']}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

# 将天气数据保存到数据库
def save_weather_to_database(city_name, weather_data):
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        sql = '''
            INSERT INTO weatherdata7 (
                城市, 观测时间, 最高温度, 最低温度, 白天天气状况,晚间天气状况,
                白天风力, 夜间风力, 降水量, 紫外线, 湿度, 能见度, 云量
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''

        for day_data in weather_data:
            values = (
                city_name,
                day_data["fxDate"],
                day_data["tempMax"],
                day_data["tempMin"],
                day_data["textDay"],
                day_data["textNight"],
                day_data["windScaleDay"],
                day_data["windScaleNight"],
                day_data["precip"],
                day_data["uvIndex"],
                day_data["humidity"],
                day_data["vis"],
                day_data["cloud"]
            )
            cursor.execute(sql, values)

        connection.commit()
        print(f"天气数据保存成功: {city_name}")

    except pymysql.Error as err:
        print(f"数据库错误: {err}")

    finally:
        if connection:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    # 读取中国城市编码数据
    china_city_codes = read_china_city_codes()

    if china_city_codes:
        for city_name, city_id in china_city_codes.items():
            # 获取7天天气信息
            weather_data_7day = crawl_7day_weather(city_id)

            if weather_data_7day:
                # 将天气数据保存到数据库
                save_weather_to_database(city_name, weather_data_7day)
