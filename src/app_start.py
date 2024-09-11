# Flask 모듈 가져오기
from flask import Flask
from flask_cors import CORS
import threading
import time
import csv
from datetime import datetime
import KBO_crawl

app = Flask(__name__)  # Flask 객체 생성
CORS(app)
from routing import *  # app 활성화 이후 app.route 목록 import

# 상태 파일 경로
DAILY_CHECK = 'csv/daily_check.csv'


# CSV 파일에서 시간 값을 읽어오는 함수
def read_time_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # 첫 번째 행에서 시간 값을 읽어온다고 가정
        for row in reader:
            return datetime.fromisoformat(row[0])


# 현재 시간을 읽어와서 비교하고 특정 함수를 실행하는 함수
def check_time(file_path):
    target_time = read_time_from_csv(file_path)
    while True:
        now = datetime.now()
        if now >= target_time:
            print("Time reached!")
            # 특정 함수 실행
            specific_function()
            # 목표 시간이 지나면 새로운 목표 시간을 CSV에서 읽어옵니다.
            target_time = read_time_from_csv(file_path)
        time.sleep(5)  # 5초 대기


# 특정 함수를 정의합니다.
def specific_function():
    print("Executing specific function!")


# 1) 오전 6시 이후에 파일 초기화
def reset_check_if_needed():
    now = datetime.datetime.now()
    if now.hour < 6:  # 오전 6시 이전에는 초기화하지 않음
        return
    if os.path.exists(DAILY_CHECK):
        df = pd.read_csv(DAILY_CHECK)
        if not df.empty:
            last_run_date = pd.to_datetime(df.iloc[-1]['date']).date()
            if last_run_date < datetime.date.today():
                # 오늘이 아닌 날짜가 마지막 실행일인 경우 상태 초기화
                df = pd.DataFrame(columns=['date'])
                df.to_csv(DAILY_CHECK, index=False)


# 2) 오늘 작업이 실행되었는지 확인
def daily_check():
    if not os.path.exists(DAILY_CHECK):
        return False
    df = pd.read_csv(DAILY_CHECK)
    if df.empty:
        return False
    today = datetime.date.today()
    last_run_date = pd.to_datetime(df.iloc[-1]['date']).date()
    return last_run_date == today


# 3) 작업 모두 종료 후 시간 기록
def record_time():
    today = datetime.datetime.now()
    if os.path.exists(DAILY_CHECK):
        df = pd.read_csv(DAILY_CHECK)
    else:
        df = pd.DataFrame(columns=['date'])
    df = pd.concat([df, pd.DataFrame([{'date': today}])], ignore_index=True)
    df.to_csv(DAILY_CHECK, index=False)


# Flask 웹 실행
if __name__ == "__main__":
    app.run(debug=True)

# 콘솔 확인
# Flask port: 5000 -> http://localhost:5000
# 테스트 1: 크롬 주소창, 테스트 2: Talend API, 테스트 3: JS AJAX
