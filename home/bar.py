from .utils import *

def highest_wind_humidity():
    try:
        # 查询风速最高的十个城市和对应的风速
        cursor.execute("SELECT 城市, 风力等级 FROM weatherdata ORDER BY 风力等级 DESC LIMIT 10")
        highest_wind = cursor.fetchall()

        # 查询湿度最高的十个城市和对应的湿度
        cursor.execute("SELECT 城市, 湿度 FROM weatherdata ORDER BY 湿度 DESC LIMIT 10")
        highest_humidity = cursor.fetchall()

    except Exception as e:
        print("查询错误:", e)
        return None

    return highest_wind, highest_humidity


# 调用函数并打印结果
highest_wind, highest_humidity = highest_wind_humidity()
# print("风速最高的十个城市:", highest_wind)
# print("湿度最高的十个城市:", highest_humidity)
