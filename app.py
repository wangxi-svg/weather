import json
import pandas as pd
from datetime import datetime, timedelta

from flask import Flask, render_template, request, redirect, session, jsonify

from home.bar import highest_wind_humidity
from home.biaoqian import count_weather
from home.line import highest_lowest_temperature
from lishi.search import get_last_n_records, search_weather
from map.utils import city_tem
from search.line import line
from search.table import table
from userUtils.query import query
from model.readModel import load_artifacts, preprocess_input, predict_weather

app = Flask(__name__)

# 设置密钥
app.secret_key = 'your_secret_key'


@app.route('/')
def every():
    return render_template('login.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        request.form = dict(request.form)

        email = request.form.get('email')
        password = request.form.get('password')

        user = query('SELECT * FROM users WHERE email = %s AND password = %s', [email, password], 'select_one')

        if user:
            session['email'] = email
            return redirect('/home', 301)

        else:
            error_message = '账号或密码错误'
            return render_template('login.html', error_message=error_message)

    else:
        return render_template('login.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        request.form = dict(request.form)
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        password_checked = request.form.get('passwordChecked')

        if password != password_checked:
            error_message = '两次密码不符'
            return render_template('register.html', error_message=error_message)

        email_exists = query('SELECT * FROM users WHERE email = %s', [email], 'select_one')
        if email_exists:
            error_message = '该邮箱已被注册'
            return render_template('register.html', error_message=error_message)

        user_exists = query('SELECT * FROM users WHERE username = %s', [username], 'select_one')
        if user_exists:
            error_message = '用户名已被注册'
            return render_template('register.html', error_message=error_message)

        query('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', [username, email, password])

        session['email'] = email
        return redirect('/login', 301)

    else:
        return render_template('register.html')


@app.route("/home")
def home():
    # 获取用户信息
    email = session.get('email')
    # 四个标签
    sunny, cloudy, rainy, snowy = count_weather()
    # 折线图
    highest_temperatures, lowest_temperatures = highest_lowest_temperature()

    # 饼图和环形图
    highest_wind, highest_humidity = highest_wind_humidity()

    return render_template('home.html',
                           email=email,
                           # 标签
                           sunny=sunny,
                           cloudy=cloudy,
                           rainy=rainy,
                           snowy=snowy,

                           # 折线图
                           highest_temperatures=highest_temperatures,
                           lowest_temperatures=lowest_temperatures,

                           # 饼图和环形图
                           highest_wind=highest_wind,
                           highest_humidity=highest_humidity
                           )


# 天气地图路由
@app.route('/map')
def map():
    # 获取用户信息
    email = session.get('email')
    temperature = city_tem()
    temperature = json.dumps(temperature)  # 将温度数据转换为 JSON 格式
    return render_template('map.html',
                           email=email,
                           temperatureData=temperature
                           )


@app.route('/search', methods=['POST', 'GET'])
def search():
    email = session.get('email')
    try:
        if request.method == 'POST':
            # 接收参数
            city = request.form.get('city')

            # 调用 line 函数获取四个不同的天气指标数据
            line_result = line(city)

            # 调用 table 函数获取天气数据表格
            table_result = table(city)

            # 将四个指标数据分别赋值给不同的变量
            highest, lowest, visibility, humidity = line_result

            # 将结果组织成字典
            search_result = {
                'highest': highest,
                'lowest': lowest,
                'visibility': visibility,
                'humidity': humidity,
                'table_result': table_result  # 将新查询的结果添加到字典中
            }
            # print("查询结果:", search_result)
            return jsonify(search_result)
        return render_template('search.html', email=email)
    except Exception as e:
        error_message = "存在错误: {}".format(str(e))
        return jsonify({"error": error_message}), 500


@app.route('/lishi', methods=['POST', 'GET'])
def lishi():
    email = session.get('email')
    try:
        if request.method == 'POST':
            city = request.form.get('city')
            date = request.form.get('date')  # 接收日期参数
            search_result = search_weather(city, date)
            # print("查询结果:", search_result)
            return jsonify(search_result)
        return render_template('lishi.html', email=email)
    except Exception as e:
        error_message = "存在错误: {}".format(str(e))
        return jsonify({"error": error_message}), 500
    

@app.route('/yuce', methods=['POST', 'GET'])
def yuce():
    email = session.get('email')
    try:
        if request.method == 'POST':
            city = request.form.get('city')
            count = 3
            # 获取历史数据
            history_data = get_last_n_records(city,count)
            if not history_data:
                return jsonify({"error": "未找到历史数据"}), 400
                
            # 按日期排序并获取最后三天的数据
            sorted_data = sorted(history_data, key=lambda x: x['日期'])
            history_data = sorted_data[-count:]
            
            if len(history_data) < count:
                return jsonify({"error": "没有足够的历史数据来进行预测"}), 400
            # 准备预测数据
            df = pd.DataFrame(history_data)
            df = df.sort_values('日期')  # 确保数据按日期排序
            print("----start")
            # 加载模型
            model, scaler, labels = load_artifacts()
            print('loadModel')
            # 准备预测
            predictions = []
            for _ in range(count):  # 预测未来三天
                print('1aaaaa')
                try:
                    # 预处理输入数据
                    features = preprocess_input(df)
                    # 进行预测
                    prediction = predict_weather(model, scaler, features)
                    predictions.append(prediction)
                    
                    # 更新数据框，添加预测结果作为新的历史数据
                    new_row = df.iloc[-1].copy()
                    new_row['日期'] = (pd.to_datetime(new_row['日期']) + timedelta(days=1)).strftime('%Y-%m-%d')
                    new_row['天气'] = prediction
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                except Exception as e:
                    # 如果预测失败，添加'N'作为预测结果
                    predictions.append('N')
                    # 仍然需要更新数据框以继续预测
                    new_row = df.iloc[-1].copy()
                    new_row['日期'] = (pd.to_datetime(new_row['日期']) + timedelta(days=1)).strftime('%Y-%m-%d')
                    new_row['天气'] = 'N'
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print('result')
            return jsonify({
                "history": history_data[-count:],  # 只返回最近三天的历史数据
                "predictions": predictions
            })
            
        return render_template('yuce.html', email=email)
    except Exception as e:
        error_message = "存在错误: {}".format(str(e))
        return jsonify({"error": error_message}), 500


# 用户管理路由
@app.route('/user', methods=['POST', 'GET'])
def user():
    email = session.get('email')
    return render_template('user.html', email=email)


@app.route('/get_user_info', methods=['POST', 'GET'])
def get_user_info():
    email = session.get('email')
    try:
        if request.method == 'GET':
            user = query('SELECT * FROM users WHERE email = %s', [email], 'select_one')
            username = user[0][1]
            useremail = user[0][2]
            usermima = user[0][3]
            print("查询结果:", user)
            # 构建返回的JSON对象
            return jsonify({
                "nickname": username,
                "email": useremail,
                "mima": usermima,
            })
    except Exception as e:
        error_message = "存在错误: {}".format(str(e))
        return jsonify({"error": error_message}), 500



@app.route('/update_user_info', methods=['POST', 'GET'])
def update_user_info():
    email = session.get('email')
    try:
        if request.method == 'POST':
            # 获取表单数据
            nickname = request.form.get('nickname')
            mima = request.form.get('mima')

            # 构建更新语句
            update_statement = '''
                UPDATE users
                SET username = %s, password = %s
                WHERE email = %s
            '''

            # 执行更新操作
            # 假设query函数可以接受一个操作类型参数，这里使用'update'
            update_result = query(update_statement, [nickname, mima, email], 'update')
            print(update_result)
            # 如果更新成功，返回成功消息
            return jsonify({
                "message": "用户信息更新成功"
            }), 200

        # GET请求的处理逻辑
        return render_template('update_user_info.html', email=email)
    except Exception as e:
        error_message = "存在错误: {}".format(str(e))
        return jsonify({"error": error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
