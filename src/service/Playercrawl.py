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
options.add_argument("--headless=new")  # GUI를 표시하지 않음
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
values = [option.get_attribute('value') for option in team_select.options if option.get_attribute('value')]
#print("Available team values:", values)
#values=['HT','SS','LG']
# 팀 선택
# for team_value in values:
#
#     # 팀 선택 드롭다운 매번 다시 찾기
#     # selenium.common.exceptions.StaleElementReferenceException 방지
#     team_select = Select(wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_ddlTeam_ddlTeam"]'))
#
#     print(f"Selecting team: {team_value}")
#     team_select.select_by_value(team_value)
#     time.sleep(1)
#
#     # 모든 <a> 태그 찾기
#     links = wd.find_elements(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/tbody/tr/td[2]/a')
#
#     # href 추출
#     hrefs = [link.get_attribute('href') for link in links if link.get_attribute('href')]
#
#     # 결과 출력
#     print("Extracted hrefs:")
#     for href in hrefs:
#         players.append(href.split("=")[1])
#         print(href)


# 각 팀의 선수 playerId 리스트
#print(players)


statList = []
newStatList=[]
names=[]

# for eachPlayer in players:
#     url_eachPlayer=f'https://www.koreabaseball.com/Record/Player/HitterDetail/Total.aspx?playerId={eachPlayer}'
#     wd.get(url_eachPlayer)
#
#     stat_elements = wd.find_elements(By.XPATH, '// *[ @ id = "contents"] / div[2] / div[2] / div / table / tbody / tr')
#     stat = [t.text for t in stat_elements if t.text]
#     for stats in stat:
#         stats.split(' ')
#         if stats.split(' ')[0]=='2023' and stats.split(' ')[2]!='-':
#             print(stats.split(' ')[4])
#             # 다 빼온 다음에
#             타석 = float(stats.split(' ')[4]); 타율=float(stats.split(' ')[2]); 홈런=float(stats.split(' ')[10]); 루타=float(stats.split(' ')[11])
#             타점 = float(stats.split(' ')[12]); 도루=float(stats.split(' ')[13]); 장타율=float(stats.split(' ')[19]); 출루율=float(stats.split(' ')[20])
#
#     # 선수 이름 추출
#     # name_elements = wd.find_elements(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_playerProfile_lblName"]')
#     # name = [n.text for n in name_elements if n.text]  # 텍스트 추출
#     # print("Name:", name)
#
#             # 급여 추출
#             salary_element = wd.find_element(By.XPATH,'//*[@id="cphContents_cphContents_cphContents_playerProfile_lblSalary"]')
#             salary_text = salary_element.text
#
#             if salary_text:  # 텍스트가 있는 경우에만 처리
#                 if salary_text.endswith('달러'):
#                     salary_value = salary_text[:-2]  # 마지막 두 글자 제거
#                     salary_value = int(salary_value) //10  # 한화로 변환
#                 else:
#                     salary_value = salary_text[:-2]  # 마지막 두 글자만 제거
#                     salary_value = int(salary_value)  # 정수로 변환
#
#                 print("Salary:", salary_value)
#                 연봉 = salary_value
#
#             statList.append([타석,타율,홈런,루타,타점,도루,장타율,출루율, 연봉])
#
#
#     # 종속변수는 연봉, 매개변수는 스탯
#     # 파이썬 day20 주택가격분석 46번째
#     print(  statList )
#
# for eachPlayer in players:
#     url_eachPlayer=f'https://www.koreabaseball.com/Record/Player/HitterDetail/Total.aspx?playerId={eachPlayer}'
#     wd.get(url_eachPlayer)
#
#     stat_elements = wd.find_elements(By.XPATH, '// *[ @ id = "contents"] / div[2] / div[2] / div / table / tbody / tr')
#     stat = [t.text for t in stat_elements if t.text]
#     for stats in stat:
#         stats.split(' ')
#         if stats.split(' ')[0]=='2024' and stats.split(' ')[2]!='-':
#             print(stats.split(' ')[4])
#             # 다 빼온 다음에
#             타석 = float(stats.split(' ')[4]); 타율=float(stats.split(' ')[2]); 홈런=float(stats.split(' ')[10]); 루타=float(stats.split(' ')[11])
#             타점 = float(stats.split(' ')[12]); 도루=float(stats.split(' ')[13]); 장타율=float(stats.split(' ')[19]); 출루율=float(stats.split(' ')[20])
#
#     #선수 이름 추출
#             name_element = wd.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_playerProfile_lblName"]')
#             name = name_element.text  # 텍스트 추출
#             names.append(name)
#             print("Name:", name)
#
#             # 급여 추출
#             salary_element = wd.find_element(By.XPATH,'//*[@id="cphContents_cphContents_cphContents_playerProfile_lblSalary"]')
#             salary_text = salary_element.text
#
#             if salary_text:  # 텍스트가 있는 경우에만 처리
#                 if salary_text.endswith('달러'):
#                     salary_value = salary_text[:-2]  # 마지막 두 글자 제거
#                     salary_value = int(salary_value) // 10  # 한화로 변환
#                 else:
#                     salary_value = salary_text[:-2]  # 마지막 두 글자만 제거
#                     salary_value = int(salary_value)  # 정수로 변환
#
#                 print("Salary:", salary_value)
#                 연봉 = salary_value
#
#             newStatList.append([타석,타율,홈런,루타,타점,도루,장타율,출루율])
#
#
#     # 종속변수는 연봉, 매개변수는 스탯
#     # 파이썬 day20 주택가격분석 46번째
#     print(  newStatList )
# ,'타율','홈런','루타','타점','도루','장타율','출루율',
# x_label=['타석' ]
# y_labe=['연봉']
# stat2023 = pd.DataFrame( statList , columns=['타석' ,'타율','홈런','루타','타점','도루','장타율','출루율', '연봉'] )
# stat2023.to_csv('stat2023.csv', index=False, encoding='utf-8-sig')  # index=False로 설정하여 인덱스를 제외
# y_2023=stat2023['연봉']
# x_2023=stat2023.drop(['연봉'],axis=1,inplace=False)
#
# print( stat2023.info())
# print( x_2023.info() )
# print( y_2023.info() )
#
#
# x_features=stat2023[x_label].values
# print( x_features )
# from sklearn.preprocessing import StandardScaler
# x_2023_s_scaled=StandardScaler().fit_transform(x_features)
# print( x_2023_s_scaled )
#
#
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x_2023_s_scaled,y_2023,test_size=0.1,random_state=156)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# y_predict=lr.predict(x_test)
# print(y_predict)
# print( y_train )
#
# mse=mean_squared_error(y_test,y_predict)
# rmse=np.sqrt(mse)
# print(f'MSE:{mse:.3f}, RMSE:{rmse:.3f}')
# print(f'R^2(Variance score): {r2_score(y_test,y_predict):.3f}')
# print('Y 절편 값: ',lr.intercept_)
# print('회귀 계수 값: ',np.round(lr.coef_,1)) # 기울기 값
#
# stat2024 = pd.DataFrame( newStatList , columns=['타석' ,'타율','홈런','루타','타점','도루','장타율','출루율'] )
#
# x_2024=stat2024
#
# x_features=stat2024[x_label].values
# print( x_features )
# x_2024_s_scaled=StandardScaler().fit_transform(x_features)
# print( x_2024_s_scaled )
#
# y_predict=lr.predict(x_2024_s_scaled)
# print(y_predict)
# print(stat2024.head())
#
# print( x_2024.info() )
# print( x_2024.head() )
#
# stat2024['선수명']=names
# stat2024.to_csv('stat2024.csv', index=False, encoding='utf-8-sig')  # index=False로 설정하여 인덱스를 제외

# df=pd.read_csv('../crawl_csv/stat2023.csv')
# print(df)


#
# x_features=stat2023[x_label].values
# print( x_features )
# from sklearn.preprocessing import StandardScaler
# x_2023_s_scaled=StandardScaler().fit_transform(x_features)
# print( x_2023_s_scaled )
#
#
#
# stat2024 = pd.DataFrame( newStatList , columns=['타석' ,'타율','홈런','루타','타점','도루','장타율','출루율', '연봉'] )
#
# # y_2024=stat2024['연봉']
# x_2024=stat2024.drop(['연봉'],axis=1,inplace=False)
#
# x_features=stat2024[x_label].values
# print( x_features )
# x_2024_s_scaled=StandardScaler().fit_transform(x_features)
# print( x_2024_s_scaled )
#
#
# from sklearn.model_selection import train_test_split
# # 3. 선형 회귀 분석 모델 생성
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# lr=LinearRegression()
# lr.fit(x_2023_s_scaled,y_2023)
# y_predict=lr.predict(x_2024_s_scaled)
# print(y_predict)










