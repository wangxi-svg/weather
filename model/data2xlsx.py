import pandas as pd
import os
import pymysql
# 数据库配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "tianqi",
}

def export_tables_to_csv():
    try:
        # 连接数据库
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        # 创建输出目录
        output_dir = "csv_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 遍历每个表并导出为CSV
        for table in tables:
            table_name = table[0]
            
            # 获取表结构信息
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            
            # 保存表结构信息
            structure_df = pd.DataFrame(columns, columns=['Field', 'Type', 'Null', 'Key', 'Default', 'Extra'])
            structure_df.to_csv(f"{output_dir}/{table_name}_structure.csv", index=False, encoding='utf-8')
            
            # 导出表数据
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # 获取列名
            column_names = [desc[0] for desc in cursor.description]
            
            # 转换为DataFrame并保存为CSV
            df = pd.DataFrame(rows, columns=column_names)
            df.to_csv(f"{output_dir}/{table_name}_data.csv", index=False, encoding='utf-8')
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"导出过程中发生错误: {str(e)}")

# 执行导出
export_tables_to_csv()
