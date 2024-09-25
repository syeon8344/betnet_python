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

if 선형회귀모델.pvalues["HLD"] < 0.5 :
    print(f"이 가설은 유효합니다 p > {선형회귀모델.pvalues["HLD"]}")
else:
    print(f"이 가설은 유효하지 않습니다 p < {선형회귀모델.pvalues["HLD"]}")

