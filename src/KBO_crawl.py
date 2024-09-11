# KBO 웹사이트 매일 1회 크롤링, 오전 6시마다 크롤링 여부 초기화
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import os
import datetime

# webdriver 객체 생성 후 크롤링 작업들 실행
wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# [1] 팀 단위 타자 성적: 팀 성적순으로 정렬된 DataFrame
def get_team_hitter_table():
    df_table_1 = None
    df_table_2 = None
    for num in range(1, 3):
        # 웹 URL
        url_hitter = f'https://www.koreabaseball.com/Record/Team/Hitter/Basic{num}.aspx'
        # 웹 페이지 열기
        wd.get(url_hitter)
        # 웹 페이지 로드를 1초 기다리기
        time.sleep(1)

        # 페이지의 소스 코드를 저장
        html = wd.page_source
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(html, "html.parser")
        # HTML 소스 코드 출력
        # print(soup.prettify())

        # DataFrame 데이터 리스트
        data = []

        # 테이블 찾기
        table = soup.find('table')
        # 테이블의 헤더와 데이터 추출
        header = [th.get_text() for th in table.find_all('th')]
        # 테이블 tr을 찾고 tr 내의 td를 처리
        rows = table.find_all('tr')
        for row in rows:
            tds = row.find_all('td')
            print(tds)
            if tds:
                # 팀 정보만 불러와서 data 리스트에 추가
                # 합계 줄 첫번째 td는 숫자가 아니므로('합계') 예외발생 -> 루프 탈출처리
                try:
                    int(tds[0].text)
                    data.append([td.get_text() for td in tds])
                # int() 예외 발생 -> 합계 행이므로 루프 탈출
                except ValueError:
                    break
        if num == 1:
            df_table_1 = pd.DataFrame(data, columns=header, index=None)
        elif num == 2:
            df_table_2 = pd.DataFrame(data, columns=header, index=None)

    # 1, 2페이지 테이블을 겹치지 않게 하나의 DataFrame으로 합치기
    df_table_2.drop(columns=['순위', '팀명', 'AVG'], inplace=True)
    df_hitter = pd.concat([df_table_1, df_table_2], axis=1)
    # print(df_hitter)
    # 데이터프레임 객체를 CSV 파일로 저장
    df_hitter.to_csv("csv/team_hitter_crawl.csv", mode='w', encoding='utf-8', index=False)
    # print(pd.read_csv("csv/team_hitter_crawl.csv", encoding='utf-8'))  # CSV 테스트


# [2] 팀 단위 투수 성적순 DataFrame 생성 및 CSV
def get_team_pitcher_table():
    df_table_1 = None
    df_table_2 = None
    for num in range(1, 3):
        # 웹 페이지 URL
        url_pitcher = f'https://www.koreabaseball.com/Record/Team/Pitcher/Basic{num}.aspx'
        # 웹 페이지 열기
        wd.get(url_pitcher)
        # 웹 페이지 로드를 1초 기다리기
        time.sleep(1)

        # 페이지의 소스 코드를 저장
        html = wd.page_source
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(html, "html.parser")
        # HTML 소스 코드 출력
        # print(soup.prettify())

        # DataFrame 데이터 리스트
        data = []
        # 테이블 찾기
        table = soup.find('table')
        # 테이블의 헤더와 데이터 추출
        header = [th.get_text() for th in table.find_all('th')]
        # 테이블 tr을 찾고 tr 내의 td를 처리
        rows = table.find_all('tr')
        for row in rows:
            tds = row.find_all('td')
            # print(tds)
            if tds:
                # 팀 정보만 불러와서 data 리스트에 추가
                # 합계 줄 첫번째 td는 숫자가 아니므로('합계') 예외발생 -> 루프 탈출처리
                try:
                    int(tds[0].text)
                    data.append([td.get_text() for td in tds])
                # int() 예외 발생 -> 합계 행이므로 루프 탈출
                except ValueError:
                    break
        if num == 1:
            df_table_1 = pd.DataFrame(data, columns=header, index=None)
        elif num == 2:
            df_table_2 = pd.DataFrame(data, columns=header, index=None)
    # 출력
    # print(data)
    # 1, 2페이지 테이블을 겹치지 않게 하나의 DataFrame으로 합치기
    df_table_2.drop(columns=['순위', '팀명', 'AVG'], inplace=True)
    df_pitcher = pd.concat([df_table_1, df_table_2], axis=1)
    # print(df_pitcher)
    # DataFrame 객체를 CSV 파일로 저장
    df_pitcher.to_csv("csv/team_pitcher_crawl.csv", mode='w', encoding='utf-8', index=False)
    # print(pd.read_csv("csv/team_pitcher_crawl.csv", encoding='utf-8'))  # CSV 테스트


# [2] 팀 단위 주루 성적순 DataFrame 생성 및 CSV
def get_team_runner_table():
    # 웹 페이지 URL
    url_runner = f'https://www.koreabaseball.com/Record/Team/Runner/Basic.aspx'
    # 웹 페이지 열기
    wd.get(url_runner)
    # 웹 페이지 로드를 1초 기다리기
    time.sleep(1)
    # 페이지의 소스 코드를 저장
    html = wd.page_source
    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html, "html.parser")
    # HTML 소스 코드 출력
    # print(soup.prettify())

    # DataFrame 데이터 리스트
    data = []
    # 테이블 찾기
    table = soup.find('table')
    # 테이블의 헤더와 데이터 추출
    header = [th.get_text() for th in table.find_all('th')]
    # 테이블 tr을 찾고 tr 내의 td를 처리
    rows = table.find_all('tr')
    for row in rows:
        tds = row.find_all('td')
        # print(tds)
        if tds:
            # 팀 정보만 불러와서 data 리스트에 추가
            # 합계 줄 첫번째 td는 숫자가 아니므로('합계') 예외발생 -> 루프 탈출처리
            try:
                int(tds[0].text)
                data.append([td.get_text() for td in tds])
            # int() 예외 발생 -> 합계 행이므로 루프 탈출
            except ValueError:
                break
    # 열 레이블 리스트와 데이터를 DataFrame으로 합치기
    df_runner = pd.DataFrame(data, columns=header, index=None)
    # print(df_runner)
    # DataFrame 객체를 CSV 파일로 저장
    df_runner.to_csv("csv/team_runner_crawl.csv", mode='w', encoding='utf-8', index=False)
    # print(pd.read_csv("csv/team_runner_crawl.csv", encoding='utf-8'))  # CSV 테스트


# cur_year = 2024
# cur_month = 8
# cur_month_str = f"{cur_month:02d}"


# TODO: 평균득실점 및 (베트맨) 맞대결공격력/수비력? 아니면 KBO사이트에서?
# 매일 경기 일정 및 선발투수 정보: KBO > 일정/결과 > 게임센터
def get_daily_data():
    # 1. 웹페이지 연결
    daily_url = "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx"
    wd.get(daily_url)
    time.sleep(1)
    # 게임센터 페이지의 경기 칸 수
    game_list = wd.find_elements(By.CLASS_NAME, 'game-cont')
    # print(len(num))  # 2024-09-06, (경기수) 4 출력
    for li in game_list:
        li.click()  # 프리뷰 탭 열림
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblStartPitcher > tbody > tr')
        # 시즌 어웨이 정보
        season_away_pitcher_name = trs[0].find_element(By.CLASS_NAME, 'name')
        season_away_pitcher_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[1:]]
        # 시즌 홈 정보
        season_home_pitcher_name = trs[1].find_element(By.CLASS_NAME, 'name')
        season_home_pitcher_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[1:]]
        # print(season_away_pitcher_name.text, home_pitcher_name.text)
        # 선발투수 홈/원정: 'attr_value' 속성값이 "HOMEAWAY"인 a 태그? -> 태그 경로 XPath 사용해서 click()
        # 크롬 개발자 도구에서 특정 태그의 XPATH 조회 가능
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[2]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblStartPitcher > tbody > tr')
        ha_away_pitcher_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[1:]]
        ha_home_pitcher_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[1:]]
        # 선발투수 맞대결
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[3]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblStartPitcher > tbody > tr')
        vs_away_pitcher_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[1:]]
        vs_home_pitcher_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[1:]]
        # 팀 전력비교 메뉴
        wd.execute_script(f"setGameDetailSection('TEAM')")  # 버튼에 할당된 자바스크립트 실행
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblRecord > tbody > tr')
        # 각 팀 이름
        away_team_name = trs[0].find_element(By.CSS_SELECTOR, 'th > span').text
        home_team_name = trs[1].find_element(By.CSS_SELECTOR, 'th > span').text
        # 팀 시즌 기록
        season_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        season_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        # 팀 홈/원정 기록
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[2]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblRecord > tbody > tr')
        ha_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        ha_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        # 팀 맞대결 기록
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[3]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblRecord > tbody > tr')
        vs_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        vs_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        print(vs_away_team_data, vs_home_team_data)
        선발투수데이터 = {
            's_era'
            's_war'
            's_match'
            's_start_avg_innings'
            's_qs'
            's_whip'
        }
        # TODO: CSV파일로 포장하기


def get_monthly_schedule(cur_year=2024, cur_month_str="09"):
    # 월간 스케줄표: 매일 갱신하면 매일 업데이트된 경기 결과도 가져올 수 있다
    monthly_url = "https://www.koreabaseball.com/Schedule/Schedule.aspx"
    wd.get(monthly_url)
    time.sleep(1)

    year_select = Select(wd.find_element(By.ID, 'ddlYear'))
    month_select = Select(wd.find_element(By.ID, 'ddlMonth'))

    # select by value
    year_select.select_by_value(str(cur_year))
    month_select.select_by_value(cur_month_str)

    monthly_html = wd.page_source
    bs = BeautifulSoup(monthly_html, 'html.parser')
    monthly_table = bs.select('table#tblScheduleList > tbody > tr')
    # print(monthly_table)
    data = []
    for tr in monthly_table:  # 테이블 행 (날짜는 각 날짜 맨 처음 행에서만 나온다)
        # print(tr)
        tds = tr.select('td')
        for td in tds:  # 행마다의 각 열 정보 쪼개기
            # print(td)
            try:
                if 'day' in td['class']:
                    month = td.string[0:2]
                    day = td.string[3:5]
                    # print("day", month, day)
                elif 'time' in td['class']:
                    hour = td.b.string
                    # print("time", time)
                elif 'play' in td['class']:
                    teams = td.select('span')
                    away_team = teams[0].string
                    home_team = teams[-1].string
                    away_score = np.nan
                    home_score = np.nan
                    if len(teams) == 5:  # 승 패 결과가 있는 경기
                        away_score = teams[1].string
                        home_score = teams[-2].string
                        # print("score", away_score, home_score)
                    # print("team", away_team, home_team)
            except KeyError:  # ['class'] 값이 day, time, play가 아니다: 원하는 값이 없으므로 continue
                continue
        # 각 tr 처리 후 추출된 데이터를 data 리스트에 dict()로 추가
        data.append({'month': month, 'day': day, 'time': hour, 'away_team': away_team, 'home_team': home_team,
                     'away_score': away_score, 'home_score': home_score})
        # print(data)
    df_monthly_schedule = pd.DataFrame(data)
    print(df_monthly_schedule)

    # CSV 파일로 저장
    df_monthly_schedule.to_csv('csv/monthly_schedule_crawl.csv', index=True, encoding="utf-8")


# def main():
#     # webdriver 객체 생성 후 크롤링 작업들 실행
#     wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
#     get_team_hitter_table()
#     get_team_pitcher_table()
#     get_team_runner_table()
#     get_daily_matches()
#     get_daily_pitchers()
#     get_daily_matchups()
#     wd.quit()  # 크롤링 종료 후 웹드라이버 닫기
#     record_time()  # 크롤링 끝나고 시간 기록

if __name__ == "__main__":
    get_daily_data()
    wd.quit()