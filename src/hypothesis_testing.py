import pandas as pd
from statsmodels.formula.api import ols

################################ 중간계투의 실력이 높을수록[홀드(HLD)가 많을수록] 승률이 더 높다. ####################################

# 데이터 수집
pitcher_pd = pd.read_csv("crawl_csv/pitcher/팀기록_투수_2024.csv", header=0, engine="python")

# HLD와 승률 추출
hld_value = pitcher_pd[["팀명", "HLD", "G", "W"]]
hld_value['승률'] = hld_value['W'] / hld_value['G']  # 승률 계산

# 승률 소수점 반올림
hld_value['승률'] = hld_value['승률'].round(3)  # 원하는 소수점 자리수로 조정

# 회귀 분석: 승률을 종속 변수, HLD를 독립 변수로 설정하여 회귀 모델을 생성
    # 종속변수는 연구자가 알고 싶어하는 결과 또는 반응
    # 독립변수는 종속변수에 영향을 주는 원인으로 작용합
회귀모형수식 = '승률 ~ HLD'
선형회귀모델 = ols(회귀모형수식, data=hld_value).fit()

# 회귀 분석 결과 출력
print(선형회귀모델.summary())
'''
OLS Regression Results                            
==============================================================================
Dep. Variable:                    승률   R-squared:                      0.502
Model:                            OLS   Adj. R-squared:                  0.440
Method:                 Least Squares   F-statistic:                     8.059
Date:                Wed, 25 Sep 2024   Prob (F-statistic):             0.0218
Time:                        17:55:11   Log-Likelihood:                 18.766
No. Observations:                  10   AIC:                            -33.53
Df Residuals:                       8   BIC:                            -32.93
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.3528      0.051      6.933      0.000       0.235       0.470
HLD            0.0020      0.001      2.839      0.022       0.000       0.004
==============================================================================
Omnibus:                        2.531   Durbin-Watson:                   2.491
Prob(Omnibus):                  0.282   Jarque-Bera (JB):                1.482
Skew:                           0.906   Prob(JB):                        0.477
Kurtosis:                       2.475   Cond. No.                         282.
==============================================================================
'''
if 선형회귀모델.pvalues["HLD"] < 0.05 :
    print(f"이 가설은 유효합니다 p > {선형회귀모델.pvalues["HLD"]}")
else:
    print(f"이 가설은 유효하지 않습니다 p < {선형회귀모델.pvalues["HLD"]}")
# 이 가설은 유효합니다 p > 0.03235630867777166

#############################################팀플레이를 많이 할수록 [희생번트(SAC)가 많을수록] 승률이 더 높다
import pandas as pd
from statsmodels.formula.api import ols

# 타자 기록 데이터 읽기
hitter_pd = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
print(hitter_pd)

hitter_pd = hitter_pd[["팀명", "SAC"]]


# 승 수(W) 데이터 읽기
win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
print(win_data)

win_data = win_data[["팀명", "경기","승"]]


# 승률 계산
win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

# 데이터 병합
merged_data = hitter_pd.merge(win_data[['팀명', '승률']], on='팀명')

# 회귀 분석: 승률을 종속 변수, SAC를 독립 변수로 설정하여 회귀 모델 생성
회귀모형수식 = '승률 ~ SAC'
선형회귀모델 = ols(회귀모형수식, data=merged_data).fit()

# 회귀 분석 결과 출력
print(선형회귀모델.summary())

# 가설 검증
if 선형회귀모델.pvalues["SAC"] < 0.05:  # p-value 기준을 0.05로 설정
    print(f"이 가설은 유효합니다. p-value: {선형회귀모델.pvalues['SAC']:.5f}")
else:
    print(f"이 가설은 유효하지 않습니다. p-value: {선형회귀모델.pvalues['SAC']:.5f}")
# 이 가설은 유효하지 않습니다. p-value: 0.29604


################################################### 최근 10경기 승률이 좋을수록 전체 승률이 더 높다

teamData = pd.read_csv("crawl_csv/rank/팀순위_2024.csv", header=0, engine="python")

teamData = teamData[["팀명", "승률","최근10경기"]]

print(teamData)
# 최근 10경기에서 승수 추출 함수
def extract_wins(record):
    return int(record.split('승')[0])  # '7승0무3패' 형식에서 승수를 추출

# 승수를 새로운 컬럼에 추가
teamData['최근10경기승'] = teamData['최근10경기'].apply(extract_wins)

# 퍼센트로 변환
teamData['최근10경기승률'] = (teamData['최근10경기승'] / 10) * 100  # 승수를 10으로 나누고 100을 곱하여 퍼센트로 변환

# 결과 출력
print(teamData[['팀명', '승률', '최근10경기', '최근10경기승률']])