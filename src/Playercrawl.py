import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 페이지 로드 상태 확인
def is_page_loaded(wd: webdriver.Chrome):
    return wd.execute_script("return document.readyState;") == "complete"

# 웹드라이버 설정
options = Options()
options.add_argument("--headless")  # GUI를 표시하지 않음
options.add_argument("--no-sandbox")  # 보안 샌드박스 비활성화

players = []

# webdriver 객체 생성
wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

url_playerhit = 'https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx'


# 웹 URL 열기
wd.get(url_playerhit)

# 페이지 로드 대기
WebDriverWait(wd, 10).until(is_page_loaded)

# 팀 선택 드롭다운 찾기
team_select = Select(wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlTeam_ddlTeam"]'))

# 모든 option의 value 값 추출 (빈 값 제외)
#values = [option.get_attribute('value') for option in team_select.options if option.get_attribute('value')]
#print("Available team values:", values)
values=['HT']
# 팀 선택
for team_value in values:

    # 팀 선택 드롭다운 매번 다시 찾기
    # selenium.common.exceptions.StaleElementReferenceException 방지
    team_select = Select(wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlTeam_ddlTeam"]'))

    print(f"Selecting team: {team_value}")
    team_select.select_by_value(team_value)
    time.sleep(1)

    # 모든 <a> 태그 찾기
    links = wd.find_elements(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/tbody/tr/td[2]/a')

    # href 추출
    hrefs = [link.get_attribute('href') for link in links if link.get_attribute('href')]

    # 결과 출력
    print("Extracted hrefs:")
    for href in hrefs:
        players.append(href.split("=")[1])
        print(href)


# 각 팀의 선수 playerId 리스트
print(players)


statList = []

for eachPlayer in players:
    url_eachPlayer=f'https://www.koreabaseball.com/Record/Player/HitterDetail/Total.aspx?playerId={eachPlayer}'
    wd.get(url_eachPlayer)

    stat_elements = wd.find_elements(By.XPATH, '// *[ @ id = "contents"] / div[2] / div[2] / div / table / tbody / tr')
    stat = [t.text for t in stat_elements if t.text]
    for stats in stat:
        stats.split(' ')
        if stats.split(' ')[0]=='2023' and stats.split(' ')[2]!='-':
            print(stats.split(' ')[4])
            # 다 빼온 다음에
            타석 = stats.split(' ')[4]

    # 선수 이름 추출
    # name_elements = wd.find_elements(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_playerProfile_lblName"]')
    # name = [n.text for n in name_elements if n.text]  # 텍스트 추출
    # print("Name:", name)

            # 급여 추출
            salary_elements = wd.find_elements(By.XPATH,
                                               '//*[@id="cphContents_cphContents_cphContents_playerProfile_lblSalary"]')
            salary = [s.text for s in salary_elements if s.text]  # 텍스트 추출
            if '달러' in salary[0]:
                salary[0]=salary[0][:-2]
                salary[0]=int(salary[0])
                salary[0]=salary[0]*100
                salary[0]=str(salary[0])+'만원'
            print("Salary:", salary)
            연봉 = salary[0]

            statList.append( [타석,연봉])


    # 종속변수는 연봉, 매개변수는 스탯
    # 파이썬 day20 주택가격분석 46번째
    print(  statList )


pd.DataFrame( statList , columns=['PA' , 'salary'] )







