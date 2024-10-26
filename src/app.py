from flask import Flask
from flask_cors import CORS
import os
import datetime
from KBO_crawl import do_crawl  # KBO_crawl.py에서 전체 크롤링 함수
from apscheduler.schedulers.background import BackgroundScheduler  # 작업 스케줄러(오전 6시 크롤링 작업)
from apscheduler.triggers.cron import CronTrigger  # 매일 특정 시간에 작업을 수행하게 하는 'cron' 트리거

app = Flask(__name__)  # Flask 객체 생성
CORS(app)
from routing import *  # app 활성화 이후 app.route 목록 import
from controller.SurveyController import *  # app 활성화 이후 SurveyController import
from controller.KBOarticle_controller import *
from controller.Salary import *
from controller.gemini import *
from hypothesis_testing import *

# 상태 파일 경로: 마지막으로 앱이 실행된 시간 기록
CRAWL_LATEST = 'crawl_csv/crawl_latest.csv'

"""
    크롤링 스케줄링: 1. 서버 실행 중 오전 6시가 되었을 시 크롤링 실행 및 날짜와 시간 저장
                   2. 오전 6시 이후 서버 실행시 오늘 크롤링 여부 체크 및 크롤링 실행
"""
# 1. 스케줄러 객체, 매일 오전 6시가 되면 크롤링 시행
scheduler = BackgroundScheduler()
scheduler.add_job(do_crawl, trigger=CronTrigger(hour=6, minute=0))  # 매일 오전 6시 정각에 do_crawl 실행


# 2. 서버 실행시 6시 이후면 크롤링 여부 확인
def check_crawl_time():
    now = datetime.datetime.now()
    if now.hour < 6:  # 오전 6시 이전에는 확인하지 않음
        print("서버가 오전 6시 이전에 실행되었으므로 스케줄링된 크롤링을 기다립니다.")
        return
    if not os.path.exists(CRAWL_LATEST):  # 파일이 존재하지 않으면 전체 크롤링 후 파일 생성
        print("크롤링 날짜 파일이 없으므로 크롤링 수행 후 새 파일을 생성합니다.")
        do_crawl(include_old_data=True)
        return
    with open(CRAWL_LATEST, 'r') as file:  # 파일에서 마지막 크롤링 날짜 읽고 비교하기
        # 첫 번째 줄 (KBO_crawl의 record_time(): 날짜만 기록)
        line = file.readline().strip()
        if not line:  # 크롤링 중 오류 등으로 인해 파일이 있지만 비어 있는 경우 크롤링 재시도
            do_crawl(include_old_data=True)
            return
        # 파일에서 읽어들인 날짜 문자열을 datetime 객체로 변환
        latest_crawl_date = datetime.datetime.strptime(line, '%Y-%m-%d').date()
        today_date = now.date()
        # 오늘 날짜와 마지막 크롤링 날짜 비교
        if today_date > latest_crawl_date:
            print("오늘 크롤링이 실행되지 않았으므로 크롤링 수행 후 날짜를 기록합니다.")
            do_crawl(include_old_data=False)
        else:
            print("오늘의 크롤링이 이미 실행되었습니다.")


# Flask 웹 실행
if __name__ == "__main__":
    check_crawl_time()
    scheduler.start()
    print("크롤링 스케줄러 시작, 오전 6시에 크롤링을 자동으로 실행합니다.")
    # 플라스크 서버 실행 app.run()은 블로킹 함수이므로 서버 종료시까지 코드가 진행되지 않는다 -> 스케줄러도 계속 실행중
    # 블로킹 함수: 프로그램 실행 중 특정 함수 또는 작업이 완료될 때까지 코드의 흐름을 멈추게 하는 호출
    app.run()