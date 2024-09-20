# 크롤링한 CSV에서 각종 지표를 이용하여 여러 승률을 계산한다.
"""
    betman.co.kr 승률 요소: 시즌승률, OPS, ERA, 득실점, 시즌공격력, 맞대결공격력, 맞대결수비력, 배당률
        - 시즌승률: 각 팀의 시즌 승/패
        - OPS: 각 팀의 최근 5경기의 출루율 및 안타율
        - ERA: 각 팀의 시즌 평균 득점값
        - 득실점: 상대팀과 최근 5경기 맞대결 이력
        - 시즌공격력: 상대팀과 최근 5경기 맞대결의 평균 안타, 평균 홈런, 총타점
        - 맞대결공격력: 홈 경기시의 홈팀 승률 OPS, 홈런, 도루, 안타, 득점
        - 맞대결수비력: 각 팀의 현재 순위 비교
        - 배당률: 현재 배당률
"""
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


"""
    타자 
    AVG: 타율, G: 경기 수, PA: 타석 수, AB: 타수, R: 득점, H: 안타, 2B: 2루타, 3B: 3루타, HR: 홈런, TB: 총루타, RBI: 타점, 
    SAC: 희생번트, SF: 희생플라이, BB: 볼넷, IBB: 고의사구, HBP: 몸에 맞는 볼, SO: 삼진, GDP: 병살타, SLG: 장타율, OBP: 출루율, 
    OPS: 출루율 + 장타율, MH: 병살타, RISP: 득점권타율, PH-BA: 대타타율
    
    투수
    ERA: 자책점 평균, G: 경기 수, W: 승리, L: 패배, SV: 세이브, HLD: 홀드, WPCT: 승률, IP: 이닝 수, H: 안타, HR: 홈런, 
    BB: 볼넷, HBP: 몸에 맞는 볼, SO: 삼진, R: 실점, ER: 자책점, WHIP: 이닝당 출루허용률, CG: 완투, SHO: 완봉, QS: 퀄리티 스타트,
    BSV: 블로우 세이브, TBF: 상대 타자 수, NP: 투구 수, 2B: 2루타, 3B: 3루타, SAC: 희생번트, SF: 희생플라이, IBB: 고의사구, 
    WP: 폭투, BK: 보크
    
    주루 
    G: 경기 수, SBA: 도루 시도, SB: 도루 성공, CS: 도루 실패, SB%: 도루 성공률, OOB: 아웃된 주자 수, PKO: 포수 견제 아웃
"""



# TODO: 타자/투수/주루를 독립변수로, 팀 순위를 종속변수로 해서 팀 순위를 예측해서 승패 예측에 사용
# [1] 데이터 준비
# 1. CSV 파일들 모으기
hitter_csv_list = glob.glob("crawl_csv/hitter/*.csv")
pitcher_csv_list = glob.glob("crawl_csv/pitcher/*.csv")
runner_csv_list = glob.glob("crawl_csv/runner/*.csv")
rank_csv_list = glob.glob("crawl_csv/rank/*.csv")

csv_per_year = list(zip(hitter_csv_list, pitcher_csv_list, runner_csv_list, rank_csv_list))
print(csv_per_year[0])
for hitter, pitcher, runner, rank in csv_per_year:
    print(hitter, pitcher, runner, rank)
    df_hitter = pd.read_csv(hitter, encoding="utf-8")[["팀명", "AVG", "R", "HR", "RBI", "BB", "SO", "SLG", "OBP", "OPS"]]
    df_pitcher = pd.read_csv(pitcher, encoding="utf-8")[["팀명", "ERA", "SV", "HLD", "IP", "SO", "WHIP", "CG", "QS", "K/BB"]]
    df_runner = pd.read_csv(runner, encoding="utf-8")[["팀명", "SBA", "SB", "CS", "SB%", "OOB", "PKO"]]
    df_rank = pd.read_csv(rank, encoding="utf-8")[["순위", "팀명", "승률"]]
    df_hitter.set_index("팀명", inplace=True)
    df_pitcher.set_index("팀명", inplace=True)
    df_runner.set_index("팀명", inplace=True)
    df_rank.set_index("팀명", inplace=True)
    concat_df = pd.concat([df_hitter, df_pitcher, df_runner, df_rank], axis=1)
    yearly_df = concat_df.sort_values(by="순위", ).reset_index()
    print(yearly_df)
# 2. 쓰지 않을 데이터열 .drop()
# df.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=True)d
# print(df.shape)
# print(df.info())
# '''
# RangeIndex: 398 entries, 0 to 397
# Data columns (total 6 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    int64
#  2   displacement  398 non-null    float64
#  3   weight        398 non-null    int64
#  4   acceleration  398 non-null    float64
#  5   model_year    398 non-null    int64
# dtypes: float64(3), int64(3)
# memory usage: 18.8 KB
# None
# '''

# # [2] 선형 회귀 모델
# # 독립 변수와 종속 변수 분리
# y = df['mpg']
# x = df.drop(['mpg'], axis=1, inplace=False)
# # 훈련용, 테스트용 데이터 분리
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# # 선형 회귀 분석 모델
# lr = LinearRegression()
# # 선형 회귀 분석 모델 훈련
# lr.fit(x_train, y_train)
# # 테스트 데이터에 대한 예측 수행: y_predict
# y_predict = lr.predict(x_test)
#
# # [3] 훈련된 선형 회귀 분석 모델 평가
# mse = mean_squared_error(y_test, y_predict)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_predict)
# print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}")
# print(f"y 절편: {np.round(lr.intercept_, 2)}, 회귀계수: {np.round(lr.coef_, 2)}")
# coef = pd.Series(data=np.round(lr.coef_, 2), index=x.columns)
# coef.sort_values(ascending=False)
# print(coef)
#
# # [4] 결과 시각화
# fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)
# x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']
# # 그래프 색상
# plot_color = ['r', 'b', 'y', 'g', 'r']
# for i, feature in enumerate(x_features):
#     row, col = i // 3, i % 3
#     sns.regplot(x=feature, y='mpg', data=df, ax=axs[row][col], color=plot_color[i])
# plt.show()
#
# # [5] 완성된 연비 예측 모델 사용
# print("연비를 예측할 차의 정보를 입력해 주세요.")
# cylinders_1 = float(input("cylinders: "))
# displacement_1 = float(input("displacement: "))
# weight_1 = float(input("weight: "))
# acceleration_1 = float(input("acceleration: "))
# model_year_1 = int(input("model_year(연식): "))
# # lr.predict()는 2차원 배열을 입력받으므로 2차원 리스트 형식으로 입력 (여러 데이터를 입력받음 -> 한 데이터마다 독립 변수 값이 여러 개 있음)
# predict_mpg = lr.predict([[cylinders_1, displacement_1, weight_1, acceleration_1, model_year_1]])
# print(f"이 차량의 예상 연비는 {predict_mpg[0]:.2f} MPG 입니다.")
#
# # + 결론 및 제언, 한계점
