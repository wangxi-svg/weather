from pymysql import connect

# 设置数据库连接信息
conn = connect(host='localhost', user='root', password='123456', database='tianqi', port=3306, charset='utf8')
cursor = conn.cursor()

def city_tem():
    try:
        # 查询所有城市的名称和温度
        cursor.execute("SELECT 城市, 温度 FROM weatherdata")

        # 获取查询结果
        city_temperature_data = cursor.fetchall()

        # 手动创建省份城市的列表
        province_cities = {
            '北京': '北京',
            '天津': '天津',
            '河北': '石家庄',
            '山西': '太原',
            '内蒙古': '呼和浩特',
            '辽宁': '沈阳',
            '吉林': '长春',
            '黑龙江': '哈尔滨',
            '上海': '上海',
            '江苏': '南京',
            '浙江': '杭州',
            '安徽': '合肥',
            '福建': '福州',
            '江西': '南昌',
            '山东': '济南',
            '河南': '郑州',
            '湖北': '武汉',
            '湖南': '长沙',
            '广东': '广州',
            '广西': '南宁',
            '海南': '海口',
            '重庆': '重庆',
            '四川': '成都',
            '贵州': '贵阳',
            '云南': '昆明',
            '西藏': '拉萨',
            '陕西': '西安',
            '甘肃': '兰州',
            '青海': '西宁',
            '宁夏': '银川',
            '新疆': '乌鲁木齐',
            '台湾': '台北',
            '香港': '香港',
            '澳门': '澳门',
        }

        # 创建空字典来存放省份温度数据
        province_temperature_data = {}

        # 将温度数据赋值给对应的省份
        for province, cities in province_cities.items():
            for city, temperature in city_temperature_data:
                if city in cities:
                    province_temperature_data[province] = temperature
                    break  # 找到城市温度后直接跳出内层循环

        # 返回按省份组织的温度数据
        return province_temperature_data
    except Exception as e:
        print("查询错误:", e)
        return None

# 示例用法：获取所有城市的天气数据
city_temperature_data = city_tem()

# 打印结果
print(city_temperature_data)
