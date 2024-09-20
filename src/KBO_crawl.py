# KBO 웹사이트 매일 1회 크롤링, 오전 6시마다 크롤링 여부 초기화
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import csv
import datetime
import time
import pandas as pd

# 상태 파일 경로: 마지막으로 앱이 실행된 시간 기록
CRAWL_LATEST = 'crawl_csv/crawl_latest.csv'


# 페이지 로드 상태 확인: 웹페이지 로드 오류시 CSV 파일 수정하지 않고 False 반환하도록
def is_page_loaded(wd: webdriver.chrome):
    return wd.execute_script("return document.readyState;") == "complete"


# [1] 팀 단위 타자 성적: 타율순으로 정렬, include_old_data=True시 2015년 데이터부터 크롤링
def get_team_hitter_table(wd: webdriver.chrome, include_old_data=False):
    # 연도를 매개변수로 받아 크롤링
    def table_crawl(year=2024):
        for num in range(1, 3):
            # 웹 URL
            url_hitter = f'https://www.koreabaseball.com/Record/Team/Hitter/Basic{num}.aspx'
            try:
                # 웹 URL 열기
                wd.get(url_hitter)
                # 페이지가 완전히 로드될 때까지 10초 기다리기
                WebDriverWait(wd, 10).until(is_page_loaded)
            except Exception as e:
                print(f"웹 페이지 접속시 오류 발생: {e}")
                raise Exception
            # 연도 선택
            year_select = Select(
                wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlSeason_ddlSeason"]'))
            # select by value
            year_select.select_by_value(str(year))
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

        # 1, 2페이지 테이블을 겹치지 않게 하나의 DataFrame으로 합치기
        df_table_2.drop(columns=['순위', '팀명', 'AVG'], inplace=True)
        df_hitter = pd.concat([df_table_1, df_table_2], axis=1)
        # print(df_hitter)
        # 데이터프레임 객체를 CSV 파일로 저장
        df_hitter.to_csv(f"crawl_csv/hitter/팀기록_타자_{year}.csv", mode='w', encoding='utf-8', index=False)

    if include_old_data:
        # 불린 True값일 시 2015년부터 2024년까지의 데이터 크롤링
        for i in range(2015, 2025):
            table_crawl(i)
        print("2015년부터의 팀 타자 기록 크롤링 성공.")
    else:
        table_crawl()
        print("팀 타자 기록 크롤링 성공.")


# [2] 팀 단위 투수 평균자책점순 DataFrame 생성 및 CSV, include_old_data=True시 2015년 데이터부터 크롤링
def get_team_pitcher_table(wd: webdriver.chrome, include_old_data=False):
    # 연도를 매개변수로 받아 크롤링
    def table_crawl(year=2024):
        for num in range(1, 3):
            # 웹 페이지 URL
            url_pitcher = f'https://www.koreabaseball.com/Record/Team/Pitcher/Basic{num}.aspx'
            try:
                # 웹 URL 열기
                wd.get(url_pitcher)
                # 페이지가 완전히 로드될 때까지 10초 기다리기
                WebDriverWait(wd, 10).until(is_page_loaded)
            except Exception as e:
                print(f"웹 페이지 접속시 오류 발생: {e}")
                raise Exception
            # 연도 선택
            year_select = Select(
                wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlSeason_ddlSeason"]'))
            # select by value
            year_select.select_by_value(str(year))
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
        df_table_2.drop(columns=['순위', '팀명', 'ERA'], inplace=True)
        df_pitcher = pd.concat([df_table_1, df_table_2], axis=1)
        # print(df_pitcher)
        # DataFrame 객체를 CSV 파일로 저장
        df_pitcher.to_csv(f"crawl_csv/pitcher/팀기록_투수_{year}.csv", mode='w', encoding='utf-8', index=False)

    if include_old_data:
        # 불린 True값일 시 2015년부터 2024년까지의 데이터 크롤링
        for i in range(2015, 2025):
            table_crawl(i)
        print("2015년부터의 팀 투수 기록 크롤링 성공.")
    else:
        table_crawl()
        print("팀 투수 기록 크롤링 성공.")


# [2] 팀 단위 주루 도루허용순 DataFrame 생성 및 CSV, include_old_data=True시 2015년 데이터부터 크롤링
def get_team_runner_table(wd: webdriver.chrome, include_old_data=False):
    # 연도를 매개변수로 받아 크롤링
    def table_crawl(year=2024):
        # 웹 페이지 URL
        url_runner = f'https://www.koreabaseball.com/Record/Team/Runner/Basic.aspx'
        try:
            # 웹 URL 열기
            wd.get(url_runner)
            # 페이지가 완전히 로드될 때까지 10초 기다리기
            WebDriverWait(wd, 10).until(is_page_loaded)
        except Exception as e:
            print(f"웹 페이지 접속시 오류 발생: {e}")
            raise Exception
        # 연도 선택
        year_select = Select(
            wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlSeason_ddlSeason"]'))
        # select by value
        year_select.select_by_value(str(year))
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
        df_runner.to_csv(f"crawl_csv/runner/팀기록_주루_{year}.csv", mode='w', encoding='utf-8', index=False)

    if include_old_data:
        # 불린 True값일 시 2015년부터 2024년까지의 데이터 크롤링
        for i in range(2015, 2025):
            table_crawl(i)
        print("2015년부터의 팀 주루 기록 크롤링 성공.")
    else:
        table_crawl()
        print("팀 주루 기록 크롤링 성공.")


# 매일 경기 일정 및 선발투수 정보: KBO > 일정/결과 > 게임센터
# -> 웹사이트에 출력용
def get_daily_data(wd: webdriver.chrome):
    # 1. 웹페이지 연결
    daily_url = "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx"
    try:
        wd.get(daily_url)
        # 페이지가 완전히 로드될 때까지 10초 기다리기
        WebDriverWait(wd, 10).until(is_page_loaded)
    except Exception as e:
        print(f"웹 페이지 접속시 오류 발생: {e}")
        raise Exception
    time.sleep(1)
    # 2. 날짜 가져오기
    date_string = wd.find_element(By.XPATH, '//*[@id="lblGameDate"]').text
    date_formatted = date_string[:10].replace('.', '-')  # 2024.09.06(요일) -> 2024-09-06
    # 게임센터 페이지의 경기 칸 수
    game_list = wd.find_elements(By.CLASS_NAME, 'game-cont')
    # print(len(num))  # 2024-09-06, (경기수) 4 출력
    # 데이터프레임 열 이름 목록
    columns_pitcher = ['일자', '홈/어웨이', '팀명', '선발투수', '시즌평균자책점', '시즌WAR', '시즌경기', '시즌선발평균이닝', '시즌QS', '시즌WHIP',
                       '홈어웨이평균자책점', '홈어웨이경기', '홈어웨이선발평균이닝', '홈어웨이QS', '홈어웨이WHIP',
                       '맞대결평균자책점', '맞대결경기', '맞대결선발평균이닝', '맞대결QS', '맞대결WHIP']
    columns_team = ['일자', '홈/어웨이', '팀명', '시즌평균자책점', '시즌타율', '시즌평균득점', '시즌평균실점', '홈어웨이평균자책점',
                    '홈어웨이시즌타율', '홈어웨이평균득점', '홈어웨이평균실점', '맞대결평균자책점', '맞대결타율', '맞대결평균득점',
                    '맞대결평균실점']
    # 최종 데이터프레임 행 목록
    data_pitcher = []
    data_team = []
    for li in game_list:
        li.click()  # 프리뷰 탭 열림

        # 우천취소 등으로 취소된 경기칸은 패스
        try:
            trs = wd.find_elements(By.CSS_SELECTOR, '#tblStartPitcher > tbody > tr')
            # 시즌 어웨이 정보
            # data: 평균자책점, WAR, 경기, 선발평균이닝, QS, WHIP
            season_away_pitcher_name = trs[0].find_element(By.CLASS_NAME, 'name').text
            season_away_pitcher_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[1:]]
        except IndexError as e:  # 취소된 경기는 프리뷰 테이블이 나오지 않아 IndexError 발생하므로 continue
            continue
        # 시즌 홈 정보
        # data: 평균자책점, WAR, 경기, 선발평균이닝, QS, WHIP
        season_home_pitcher_name = trs[1].find_element(By.CLASS_NAME, 'name').text
        season_home_pitcher_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[1:]]
        # print(season_away_pitcher_name.text, home_pitcher_name.text)
        # 선발투수 홈/원정: 'attr_value' 속성값이 "HOMEAWAY"인 a 태그? -> 태그 경로 XPath 사용해서 click()
        # 크롬 개발자 도구에서 특정 태그의 XPATH 조회 가능
        # data: 평균자책점, 경기, 선발평균이닝, QS, WHIP
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[2]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblStartPitcher > tbody > tr')
        ha_away_pitcher_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[1:]]
        ha_home_pitcher_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[1:]]
        # 선발투수 맞대결
        # data: 평균자책점, 경기, 선발평균이닝, QS, WHIP
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
        # data: 평균자책점 타율 평균득점 평균실점
        season_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        season_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        # 팀 홈/원정 기록
        # data: 평균자책점 타율 평균득점 평균실점
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[2]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblRecord > tbody > tr')
        ha_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        ha_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        # 팀 맞대결 기록
        # data: 평균자책점 타율 평균득점 평균실점
        wd.find_element(By.XPATH, '//*[@id="gameCenterContents"]/div[2]/ul/li[3]/a').click()
        trs = wd.find_elements(By.CSS_SELECTOR, '#tblRecord > tbody > tr')
        vs_away_team_data = [td.text for td in trs[0].find_elements(By.TAG_NAME, 'td')[2:]]
        vs_home_team_data = [td.text for td in trs[1].find_elements(By.TAG_NAME, 'td')[2:]]
        # print(vs_away_team_data, vs_home_team_data)
        # 데이터프레임 행 구성
        pitchers_data_away = ([date_formatted, "어웨이", away_team_name, season_away_pitcher_name] +
                              season_away_pitcher_data + ha_away_pitcher_data + vs_away_pitcher_data)
        pitchers_data_home = ([date_formatted, "홈", home_team_name, season_home_pitcher_name] +
                              season_home_pitcher_data + ha_home_pitcher_data + vs_home_pitcher_data)
        team_data_away = ([date_formatted, "어웨이", away_team_name] + season_away_team_data + ha_away_team_data +
                          vs_away_team_data)
        team_data_home = ([date_formatted, "홈", home_team_name] + season_home_team_data +
                          ha_home_team_data + vs_home_team_data)
        # 최종 데이터프레임 데이터 리스트에 각 행 추가
        data_pitcher.append(pitchers_data_away)
        data_pitcher.append(pitchers_data_home)
        data_team.append(team_data_away)
        data_team.append(team_data_home)
    # li 요소 순환 완료 후 완성된 데이터프레임 생성 및 CSV 저장
    today = datetime.datetime.today().date()
    df_pitcher = pd.DataFrame(data_pitcher, columns=columns_pitcher)
    df_pitcher.to_csv(f"crawl_csv/일일경기선발투수정보_{today}.csv", index=False, encoding="utf-8")
    df_team = pd.DataFrame(data_team, columns=columns_team)
    df_team.to_csv(f"crawl_csv/일일경기팀정보_{today}.csv", index=False, encoding="utf-8")
    print("일일 경기 정보 크롤링 성공.")


# 월간 스케줄표: 매일 갱신하면 매일 업데이트된 경기 결과도 가져올 수 있다.
"""
    인덱스 문제: 우천 등으로 경기가 취소되는 경우 새로운 경기가 나중에 추가되면 미래 경기에 미리 배팅할 때 인덱스가 깨진다.
    -> 경기 일정이 월마다 미리 등록되어 있으므로 취소된 경기가 나중에 일정표에 추가되면 데이터프레임 맨 끝으로 보내 인덱스 유지
"""


# TODO: 승/패 확률을 미리 DataFrame에 포함해서 저장하기?
def get_monthly_schedule(wd: webdriver.chrome):
    # 다른 연도 및 월 데이터 크롤링시 매개변수 추가: cur_year=2024, cur_month_str="09"

    monthly_url = "https://www.koreabaseball.com/Schedule/Schedule.aspx"
    try:
        wd.get(monthly_url)
        # 페이지가 완전히 로드될 때까지 10초 기다리기
        WebDriverWait(wd, 10).until(is_page_loaded)
    except Exception as e:
        print(f"웹 페이지 접속시 오류 발생: {e}")
        raise Exception
    time.sleep(1)
    # 연도 및 월 선택칸은 <select>
    year_select = Select(wd.find_element(By.ID, 'ddlYear'))
    selected_year = year_select.first_selected_option.text
    month_select = Select(wd.find_element(By.ID, 'ddlMonth'))
    selected_month = month_select.first_selected_option.text
    date = "".join([selected_year, selected_month])

    # 다른 연도 및 월 데이터 크롤링시 select by value
    # year_select.select_by_value(str(cur_year))
    # month_select.select_by_value(cur_month_str)

    monthly_html = wd.page_source
    bs = BeautifulSoup(monthly_html, 'html.parser')
    monthly_table = bs.select('table#tblScheduleList > tbody > tr')
    # print(monthly_table)
    data = []
    for tr in monthly_table:  # 테이블 행 (날짜는 각 날짜 맨 처음 행에서만 나온다)
        # print(tr)
        tds = tr.select('td')
        # print(tds)
        match_state = tds[-1].string
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
        # 경기고유코드: 20240901-롯데-1400, 연월일-홈팀명-시작시간
        match_code = f"{date}{day}-{home_team}-{hour.replace(':', "")}"
        data.append({'월': month, '일': day, '시작시간': hour, '어웨이팀명': away_team, '홈팀명': home_team,
                     '어웨이점수': away_score, '홈점수': home_score, '비고': match_state, '경기코드': match_code})
        # print(data)
    df_monthly_schedule = pd.DataFrame(data)
    # print(df_monthly_schedule)

    # 우천 취소 경기가 있어도 기존 경기들 인덱스가 꼬이지 않도록 설정
    # 기존 CSV 존재시 현재 크롤링된 DataFrame과 비교, 새로 추가된 경기들( = 우천 취소로 인한 새 경기 등)을 골라내 맨 뒤로
    try:
        # dtype=str: 모든 열의 타입을 str(object)로 설정, 월 열 내용 오류 방지
        df_old = pd.read_csv(f'crawl_csv/monthly_schedule/월간경기일정_{date}.csv', encoding='utf-8', dtype=str)
        # 기존 CSV의 DataFrame에 현재 DataFrame을 합치고 (인덱스 제외) 중복 행을 제거 = 새로 추가된 경기들
        new_matches = combined = pd.concat([df_old, df_monthly_schedule]).drop_duplicates(keep=False)
        df_combined = pd.concat([df_old, new_matches], ignore_index=True)
        # 인덱스 재설정 (하지 않으면 기존 인덱스가 유지된다)
        df_combined = df_combined.reset_index(drop=True)
        df_combined.to_csv(f'crawl_csv/monthly_schedule/월간경기일정_{date}.csv', encoding='utf-8', index=False)
    except FileNotFoundError:
        # 이전 CSV 파일이 없으므로 바로 저장
        df_monthly_schedule.to_csv(f'crawl_csv/monthly_schedule/월간경기일정_{date}.csv', encoding="utf-8", index=False)
    print("월간 경기 일정 크롤링 성공.")


# 연도별 팀 순위
def get_team_rank(wd: webdriver.chrome, include_old_data=False):
    # 연도를 매개변수로 받아 크롤링
    def table_crawl(year=2024):
        # 웹 페이지 URL
        url_rank = f'https://www.koreabaseball.com/Record/TeamRank/TeamRank.aspx'
        try:
            # 웹 URL 열기
            wd.get(url_rank)
            # 페이지가 완전히 로드될 때까지 10초 기다리기
            WebDriverWait(wd, 10).until(is_page_loaded)
        except Exception as e:
            print(f"웹 페이지 접속시 오류 발생: {e}")
            raise Exception
        # 연도 선택
        year_select = Select(
            wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlYear"]'))
        # select by value
        year_select.select_by_value(str(year))
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
        for row in rows[1:]:
            tds = row.find_all('td')
            # print(tds)
            # 테이블 행마다 리스트로 정보 추출
            data.append([td.get_text() for td in tds])
        # 열 레이블 리스트와 데이터를 DataFrame으로 합치기
        df_rank = pd.DataFrame(data, columns=header, index=None)
        # print(df_runner)
        # DataFrame 객체를 CSV 파일로 저장
        df_rank.to_csv(f"crawl_csv/rank/팀순위_{year}.csv", mode='w', encoding='utf-8', index=False)

    if include_old_data:
        # 불린 True값일 시 2015년부터 2024년까지의 데이터 크롤링
        for i in range(2015, 2025):
            table_crawl(i)
        print("2015년부터의 팀 순위 크롤링 성공.")
    else:
        table_crawl()
        print("팀 순위 크롤링 성공.")


# 작업 모두 종료 후 날짜 기록
def record_time():
    with open(CRAWL_LATEST, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().date()])
        print("크롤링 날짜 입력 성공.")


# 크롤링 재시도 변수
timeout_count = 0


def do_crawl(include_old_data=False):
    global timeout_count
    print("크롤링 작업을 시작합니다.")
    # webdriver 객체 생성
    options = Options()  # 웹드라이버 설정
    options.add_argument("--headless")  # 브라우저 GUI를 표시하지 않음
    options.add_argument("--no-sandbox")  # 보안 샌드박스 비활성화
    wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    try:
        # 크롤링 작업들 실행
        get_team_hitter_table(wd, include_old_data)
        get_team_pitcher_table(wd, include_old_data)
        get_team_runner_table(wd, include_old_data)
        get_daily_data(wd)
        get_monthly_schedule(wd)
        get_team_rank(wd, include_old_data)
        record_time()
        print("크롤링 작업 성공.")
        wd.quit()  # 크롤링 종료 후 웹드라이버 닫기
        timeout_count = 0  # 재시도 변수 초기화
    except Exception:  # 웹페이지 로드 오류 (서버 닫힘 등으로 인해)
        if timeout_count == 0:
            print("크롤링 작업이 실패했습니다. 다시 시도하는 중입니다.")
            wd.quit()  # 웹드라이버 닫기
            timeout_count += 1
            do_crawl()  # 한번 더 재시도
        elif timeout_count == 1:
            wd.quit()  # 웹드라이버 닫기
            print("크롤링 작업이 다시 실패했습니다. 웹페이지 오류가 있는지 확인해 주세요.")
            timeout_count = 0  # 재시도 변수 초기화
    finally:
        wd.quit()  # 크롤링 종료 후 웹드라이버 닫기


if __name__ == "__main__":
    # 파일 직접 실행시 실행되는 부분
    # webdriver 객체 생성
    options = Options()  # 웹드라이버 설정
    options.add_argument("--headless")  # 브라우저 GUI를 표시하지 않음
    options.add_argument("--no-sandbox")  # 보안 샌드박스 비활성화
    wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    # include_old_data=True일 시 2015년도 팀 타자/투수/주루 데이터부터 크롤링
    do_crawl(include_old_data=False)
    wd.quit()