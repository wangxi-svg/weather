from .utils import connect_to_database
def search_weather(city, date):
    connection = connect_to_database()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `城市`, `日期`, `最高温度`, `最低温度`, `天气`, `风向` FROM lishiweathers WHERE `城市` LIKE %s AND `日期` LIKE %s"
            city = '%' + city + '%'
            date = '%' + date + '%'  # 修改日期的模糊匹配
            cursor.execute(sql, (city, date))
            result = cursor.fetchall()
            return result
    finally:
        connection.close()

def get_last_n_records(city, n=3):
    """
    获取指定城市最近的n条天气记录
    
    Args:
        city: 城市名称
        n: 需要获取的记录数量，默认为3条
        
    Returns:
        最近n条天气记录列表
    """
    connection = connect_to_database()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT `城市`, `日期`, `最高温度`, `最低温度`, `天气`, `风向` 
                FROM lishiweathers 
                WHERE `城市` LIKE %s
                ORDER BY `日期` DESC 
                LIMIT %s
            """
            city = '%' + city + '%'
            cursor.execute(sql, (city, n))
            result = cursor.fetchall()
            return result
    finally:
        connection.close()
