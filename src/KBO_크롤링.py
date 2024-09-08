from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# day12 > 1_selenium.py
# 1. 모듈 가져오기
# from selenium에 느낌표로 또는 설정 - 프로젝트 - 인터프리터 - + 버튼에서 검색

# 2. webdriver 객체 생성
wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# 3. webdriver 객체를 이용한 웹페이지 접속, .get(URL)
# wd.get("http://hanbit.co.kr")

# 웹 페이지 URL
url_hitter = 'https://www.koreabaseball.com/Record/Team/Hitter/Basic1.aspx'

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

# 출력
print(data)

df = pd.DataFrame(data, columns=header)

# [1] 팀 단위 타자 성적: 팀 이름 순서로 정렬되고 인덱스가 초기화된 최종 DataFrame
df_hitter = df.sort_values(by='팀명').reset_index(drop=True)
print(df_hitter)




# 브라우저 종료
wd.quit()


# 데이터프레임 객체를 CSV 파일로 저장
# df.to_csv("csv/커피빈매장목록.csv", mode='w', encoding='utf-8', index=True)
