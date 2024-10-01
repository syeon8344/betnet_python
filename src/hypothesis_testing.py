import pandas as pd
from statsmodels.formula.api import ols

from flask import Flask,jsonify
app = Flask( __name__ )

from flask_cors import CORS
CORS( app ) # 모든 경로에 대해 CORS 허용




################################################################################### 중간계투의 실력이 높을수록[홀드(HLD)가 많을수록] 승률이 더 높다.
def hypoHld() :

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
    # print(선형회귀모델.summary())
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
    # 가설 검증
    if 선형회귀모델.pvalues["HLD"] < 0.05:
        print(f"중간계투의 실력이 높을수록[홀드가 많을수록] 승률이 더 높다 p-value: {선형회귀모델.pvalues['HLD']:.5f}")
    else:
        print(f"중간계투의 실력이 높다고 해도[홀드(HLD)가 많아도] 승률이 더 높은건 아니다. p-value: {선형회귀모델.pvalues['HLD']:.5f}")
        # 이 가설은 유효합니다 p > 0.03235630867777166

    # 가설 return
    return {"가설결과": "중간계투의 실력이 높을수록[홀드가 많을수록] 승률이 더 높다."}

# hypoHld()

#########################################################################################################################팀플레이를 많이 할수록 [희생번트(SAC)가 많을수록] 승률이 더 높다
def hypoSac() :
    # 타자 기록 데이터 읽기
    hitter_pd = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
    # print(hitter_pd)

    hitter_pd = hitter_pd[["팀명", "SAC"]]

    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)

    win_data = win_data[["팀명", "경기","승"]]


    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    # 데이터 병합
        # 데이터 합차기 SQL에서 inner join 같은 느낌
        # hitter.더하기(win[병합할 데이터들] on=기준으로 데이터병합 )
            # 두개의 데이터가 팀명은 공통이니까 팀명을 기준으로 합치기
    merged_data = hitter_pd.merge(win_data[['팀명', '승률']], on='팀명')

    # 회귀 분석: 승률을 종속 변수, SAC를 독립 변수로 설정하여 회귀 모델 생성
    회귀모형수식 = '승률 ~ SAC'
    선형회귀모델 = ols(회귀모형수식, data=merged_data).fit()

    # 회귀 분석 결과 출력
    # print(선형회귀모델.summary())
    '''
     OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.135
    Model:                            OLS   Adj. R-squared:                  0.027
    Method:                 Least Squares   F-statistic:                     1.250
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):              0.296
    Time:                        12:38:24   Log-Likelihood:                 15.863
    No. Observations:                  10   AIC:                            -27.73
    Df Residuals:                       8   BIC:                            -27.12
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.3693      0.112      3.311      0.011       0.112       0.627
    SAC            0.0026      0.002      1.118      0.296      -0.003       0.008
    ==============================================================================
    Omnibus:                        7.279   Durbin-Watson:                   1.622
    Prob(Omnibus):                  0.026   Jarque-Bera (JB):                2.866
    Skew:                           1.218   Prob(JB):                        0.239
    Kurtosis:                       3.970   Cond. No.                         307.
    ==============================================================================
    '''
    # 가설 검증
    if 선형회귀모델.pvalues["SAC"] < 0.05:  # p-value 기준을 0.05로 설정
        print(f"팀플레이를 많이 하는 팀일수록[희생번트 성공률이 높을수록] 승률이 더높다. p-value: {선형회귀모델.pvalues['SAC']:.5f}")
    else:
        print(f"팀플레이를 많이 하는 팀이라고[희생번트 성공률이 높다고] 승률이 더 높은건 아니다. p-value: {선형회귀모델.pvalues['SAC']:.5f}")

    return {"가설결과": "팀플레이를 많이 하는 팀이라고[희생번트 성공률이 높다고] 승률이 더 높은건 아니다."}

# hypoSac()

######################################################################################################################## 최근 10경기 승률이 좋을수록 전체 승률이 더 높다
def hypoRecent() :
    teamData = pd.read_csv("crawl_csv/rank/팀순위_2024.csv", header=0, engine="python")

    teamData = teamData[["팀명", "승률","최근10경기"]]

    # print(teamData)

    # 승수를 새로운 컬럼에 추가
        # 승 앞에 있는 숫자만 가져오기
    teamData['최근10경기승'] = teamData['최근10경기'].apply(extract_wins)

    # 퍼센트로 변환
    teamData['최근10경기승률'] = (teamData['최근10경기승'] / 10) * 100  # 승수를 10으로 나누고 100을 곱하여 퍼센트로 변환

    # 결과 출력
    # print(teamData[['팀명', '승률', '최근10경기', '최근10경기승률']])

    # 회귀 분석: 승률을 종속 변수, 최근10경기승률을 독립 변수로 설정하여 회귀 모델 생성
    회귀모형수식 = '승률 ~ 최근10경기승률'
    선형회귀모델 = ols(회귀모형수식, data=teamData).fit()

    # 회귀 분석 결과 출력
    # print(선형회귀모델.summary())
    '''
                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.390
    Model:                            OLS   Adj. R-squared:                  0.313
    Method:                 Least Squares   F-statistic:                     5.110
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):             0.0537
    Time:                        14:41:00   Log-Likelihood:                 17.387
    No. Observations:                  10   AIC:                            -30.77
    Df Residuals:                       8   BIC:                            -30.17
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.3943      0.049      8.050      0.000       0.281       0.507
    최근10경기승률       0.0021      0.001      2.260      0.054   -4.25e-05       0.004
    ==============================================================================
    Omnibus:                        0.836   Durbin-Watson:                   0.831
    Prob(Omnibus):                  0.658   Jarque-Bera (JB):                0.616
    Skew:                           0.515   Prob(JB):                        0.735
    Kurtosis:                       2.354   Cond. No.                         171.
    ==============================================================================
    '''
     # 가설 검증
    if 선형회귀모델.pvalues["최근10경기승률"] < 0.05:
        print(f"최근 10경기 승률이 높은팀이 전체 승률이 더높다. p-value: {선형회귀모델.pvalues['최근10경기승률']:.5f}")
    else:
        print(f"최근 10경기 승률이 높다고 전체적인 승률이 더 높지는 않다. p-value: {선형회귀모델.pvalues['최근10경기승률']:.5f}")

    # 이 가설은 유효하지 않습니다. p-value: 0.05369
    return {"가설결과": "최근 10경기 승률이 높다고 전체적인 승률이 더 높지는 않다."}

# 최근 10경기에서 승수 추출 함수
def extract_wins(record):
    return int(record.split('승')[0])  # '7승0무3패' 형식에서 승수를 추출

# hypoRecent()

######################################################################################################################팀의 투수의 실력이 좋을수록(ERA가 낮을수록)승률이 더 높다.
def hypoEra() :
    EraData = pd.read_csv("crawl_csv/pitcher/팀기록_투수_2024.csv", header=0, engine="python")
    # print(EraData)
    EraData = EraData[["팀명","ERA","G","W"]]
    # print(EraData)

    EraData["승률"] = EraData["W"]/EraData["G"]
    EraData["승률"] = EraData["승률"].round(2)

    # 음수로 변환 => 숫자가 낮아도 음수로 전체 바꾸면 가장 낮은수가 높은수 됨
    # 이건 다른 가설들 처럼 ~~가 높으면 승률이 높다가 아니라
    # ~~가 낮으면 이라서 음수로 변환
    EraData["ERA_minus"] = -EraData["ERA"]
    # print(EraData)


    # 회귀 분석: 승률을 종속 변수, 최근10경기승률을 독립 변수로 설정하여 회귀 모델 생성
    회귀모형수식 = '승률 ~ ERA_minus'
    선형회귀모델 = ols(회귀모형수식, data=EraData).fit()

    # 회귀 분석 결과 출력
    # print(선형회귀모델.summary())

    # 가설 검증
    if 선형회귀모델.pvalues["ERA_minus"] < 0.05:
        print(f"팀의 투수의 실력이 좋을수록(ERA가 낮을수록)승률이 더 높다. p-value: {선형회귀모델.pvalues['ERA_minus']:.5f}")
    else:
        print(f"팀의 투수의 실력이 좋다고(ERA가 낮다고)승률이 더 높은건 아니다. p-value: {선형회귀모델.pvalues['ERA_minus']:.5f}")
    #팀의 투수의 실력이 좋을수록(ERA가 낮을수록)승률이 더 높다. p-value: 0.01146
    return {"가설결과": "팀의 투수의 실력이 좋을수록(ERA가 낮을수록)승률이 더 높다."}

# hypoEra()

########################################################################################################################타자의 선구안이 좋을수록[삼진(SO)을 당한 수가 낮을수록] 승률이 더 높다.
def hypoSo() :

    so_data = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
    so_data = so_data[["팀명","SO"]]
    # print(so_data)

    ########################타자는 승률이 없어서 타자꺼 할때마다 가져오는 승률 구하기 코드
    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)

    win_data = win_data[["팀명", "경기", "승"]]

    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    # print(win_data)

    merge_data = so_data.merge(win_data[["팀명","승률"]],on="팀명")
    # print(merge_data)

    # SO를 음수로 변환
        # 이건 다른 가설들 처럼 ~~가 높으면 승률이 높다가 아니라
            # ~~가 낮으면 이라서 음수로 변환
    merge_data['SO_minus'] = -merge_data['SO']

    회귀모형수식 = '승률 ~ SO_minus'
    선형회귀모델 = ols(회귀모형수식, data=merge_data).fit()

    # print(선형회귀모델.summary())
    '''
                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.400
    Model:                            OLS   Adj. R-squared:                  0.325
    Method:                 Least Squares   F-statistic:                     5.339
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):             0.0496
    Time:                        15:11:16   Log-Likelihood:                 17.694
    No. Observations:                  10   AIC:                            -31.39
    Df Residuals:                       8   BIC:                            -30.78
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.9537      0.200      4.766      0.001       0.492       1.415
    SO_minus       0.0004      0.000      2.311      0.050    8.74e-07       0.001
    ==============================================================================
    Omnibus:                        1.053   Durbin-Watson:                   2.802
    Prob(Omnibus):                  0.591   Jarque-Bera (JB):                0.552
    Skew:                           0.538   Prob(JB):                        0.759
    Kurtosis:                       2.590   Cond. No.                     1.45e+04
    ==============================================================================
    '''
    # 가설 검증
    if 선형회귀모델.pvalues["SO_minus"] < 0.05:
        print(f"타자의 선구안이 좋을수록[삼진(SO)을 당한 수가 낮을수록] 승률이 더 높다. p-value: {선형회귀모델.pvalues['SO_minus']:.5f}")
    else:
        print(f"타자의 선구안이 좋다고[삼진(SO)을 당한 수가 낮다고] 승률이 더 높은건 아니다. p-value: {선형회귀모델.pvalues['SO_minus']:.5f}")
    #타자의 선구안이 좋을수록[삼진(SO)을 당한 수가 낮을수록] 승률이 더 높다. p-value: 0.04964
    return {"가설결과": "타자의 선구안이 좋을수록[삼진(SO)을 당한 수가 낮을수록] 승률이 더 높다."}

# hypoSo()

######################################################################################################################## 팀의 작전 성공률[대타타율(PH-BA)]이 좋을수록 승률이 더 높다.
def hypoPhBa() :

    phba_data = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
    phba_data = phba_data[["팀명","PH-BA"]]
    # print(phba_data)


    ########################타자는 승률이 없어서 타자꺼 할때마다 가져오는 승률 구하기 코드
    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)

    win_data = win_data[["팀명", "경기", "승"]]

    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    # print(win_data)

    # PH-BA 와 승률 데이터 합치기
    merge_data = phba_data.merge(win_data[["팀명","승률"]], on="팀명")
    merge_data["PH_BA"] = merge_data["PH-BA"]
    # print(merge_data)

    회귀모형수식 = '승률 ~ PH_BA'
    선형회귀모델 = ols(회귀모형수식, data=merge_data).fit()

    '''
                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.328
    Model:                            OLS   Adj. R-squared:                  0.244
    Method:                 Least Squares   F-statistic:                     3.901
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):             0.0837
    Time:                        15:25:21   Log-Likelihood:                 17.123
    No. Observations:                  10   AIC:                            -30.25
    Df Residuals:                       8   BIC:                            -29.64
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.3289      0.084      3.902      0.005       0.135       0.523
    PH_BA          0.6663      0.337      1.975      0.084      -0.112       1.444
    ==============================================================================
    Omnibus:                        2.020   Durbin-Watson:                   3.071
    Prob(Omnibus):                  0.364   Jarque-Bera (JB):                0.856
    Skew:                          -0.165   Prob(JB):                        0.652
    Kurtosis:                       1.605   Cond. No.                         23.2
    ==============================================================================
    '''
    # print(선형회귀모델.summary())

    # 가설 검증
    if 선형회귀모델.pvalues["PH_BA"] < 0.05:
        print(f"팀의 작전 성공률[대타타율(PH-BA)]이 좋을수록 승률이 더 높다. p-value: {선형회귀모델.pvalues['PH_BA']:.5f}")
    else:
        print(f"팀의 작전 성공률[대타타율(PH-BA)]이 높다고 승률이 더 높은건 아니다. p-value: {선형회귀모델.pvalues['PH_BA']:.5f}")
    # 팀의 작전 성공률[대타타율(PH-BA)]이 높다고 승률이 더 높은건 아니다. p-value: 0.08368
    return {"가설결과": "팀의 작전 성공률[대타타율(PH-BA)]이 높다고 승률이 더 높은건 아니다."}

# hypoPhBa()




########################################################################################################################투수가 공을 많이 던질수록 [투구수(NP)가 많을수록] 승률이 더 높다.
def hypoNp() :
    np_data = pd.read_csv("crawl_csv/pitcher/팀기록_투수_2024.csv", header=0, engine="python")
    np_data = np_data[["팀명","NP","G","W"]]
    # print(np_data)

    np_data["승률"] = np_data["W"] / np_data["G"]
    np_data["승률"] = np_data["승률"].round(2)
    # print(np_data)

    회귀모형수식 = "승률 ~ NP"
    선형회귀모델 = ols(회귀모형수식, data=np_data).fit()
    '''
                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.254
    Model:                            OLS   Adj. R-squared:                  0.161
    Method:                 Least Squares   F-statistic:                     2.721
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):              0.138
    Time:                        15:47:39   Log-Likelihood:                 16.663
    No. Observations:                  10   AIC:                            -29.33
    Df Residuals:                       8   BIC:                            -28.72
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.6994      0.723     -0.967      0.362      -2.367       0.968
    NP          5.482e-05   3.32e-05      1.649      0.138   -2.18e-05       0.000
    ==============================================================================
    Omnibus:                        7.413   Durbin-Watson:                   0.720
    Prob(Omnibus):                  0.025   Jarque-Bera (JB):                3.049
    Skew:                           1.283   Prob(JB):                        0.218
    Kurtosis:                       3.856   Cond. No.                     9.73e+05
    ==============================================================================
    '''
    # print(선형회귀모델.summary())

    # 가설 검증
    if 선형회귀모델.pvalues["NP"] < 0.05:
        print(f"투수가 공을 많이 던질수록 [투구수(NP)가 많을수록] 승률이 더 높다. p-value: {선형회귀모델.pvalues['NP']:.5f}")
    else:
        print(f"투수가 공을 많이 던질수록 [투구수(NP)가 많을수록] 승률이 더 낲다. p-value: {선형회귀모델.pvalues['NP']:.5f}")
    #투수가 공을 많이 던질수록 [투구수(NP)가 많을수록] 승률이 더 낲다. p-value: 0.13767
    return {"가설결과": "투수가 공을 많이 던질수록 [투구수(NP)가 많을수록] 승률이 더 높다."}

#hypoNp()



########################################################################################################################팀의 거포형 타자들이 많을수록[안타(H)보다 홈런(HR)의 비율이 많을수록] 승률이 더높다.
def hypoHr () :
    hr_data = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
    hr_data = hr_data[["팀명","H","HR"]]
    hr_data["안타홈런비율"] = hr_data['H'] / hr_data['HR']
    hr_data["안타홈런비율"] = hr_data["안타홈런비율"].round(2)
    # print(hr_data)

    ########################타자는 승률이 없어서 타자꺼 할때마다 가져오는 승률 구하기 코드
    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)
    win_data = win_data[["팀명", "경기", "승"]]
    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    merge_data = hr_data.merge(win_data[["팀명","승률"]], on="팀명")
    # print(merge_data)

    회귀모형수식 = "승률 ~ 안타홈런비율"
    선형회귀모델 = ols(회귀모형수식, data=merge_data).fit()

    # print(선형회귀모델.summary())
    '''
                               OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.129
    Model:                            OLS   Adj. R-squared:                  0.020
    Method:                 Least Squares   F-statistic:                     1.181
    Date:                Fri, 27 Sep 2024   Prob (F-statistic):              0.309
    Time:                        15:57:49   Log-Likelihood:                 15.826
    No. Observations:                  10   AIC:                            -27.65
    Df Residuals:                       8   BIC:                            -27.05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.6022      0.102      5.879      0.000       0.366       0.838
    안타홈런비율        -0.0111      0.010     -1.087      0.309      -0.035       0.012
    ==============================================================================
    Omnibus:                        2.271   Durbin-Watson:                   1.783
    Prob(Omnibus):                  0.321   Jarque-Bera (JB):                0.778
    Skew:                           0.683   Prob(JB):                        0.678
    Kurtosis:                       3.038   Cond. No.                         59.1
    ==============================================================================
    '''
    # 가설 검증
    if 선형회귀모델.pvalues["안타홈런비율"] < 0.05:
        print(f"팀의 거포형 타자들이 많을수록[안타(H)보다 홈런(HR)의 비율이 많을수록] 승률이 더높다. p-value: {선형회귀모델.pvalues['안타홈런비율']:.5f}")
    else:
        print(f"팀의 거포형 타자들이 많아도[안타(H)보다 홈런(HR)의 비율이 높아도] 승률에 영향을 끼치진 않는다. p-value: {선형회귀모델.pvalues['안타홈런비율']:.5f}")
    # 팀의 거포형 타자들이 많아도[안타(H)보다 홈런(HR)의 비율이 높아도] 승률에 영향을 끼치진 않는다. p-value: 0.30882
    return {"가설결과": "팀의 거포형 타자들이 많아도[안타(H)보다 홈런(HR)의 비율이 높아도] 승률에 영향을 끼치진 않는다."}

#hypoHr()

import scipy.stats as stats
######################################################################################################################## 홈팀이 이길 확률이 더 높다.
 # 두개 홈 승률 vs 원정 승류 을 비교하는거니까 회귀분석말고 T검증
def hypoHome () :
    home_data = pd.read_csv("crawl_csv/rank/팀순위_2024.csv", header=0, engine="python")
    home_data = home_data[["팀명","홈","방문"]]
    print(home_data)

    #######################################################(-) 제외한 홈 승률 계산
    home_data["홈"] = home_data["홈"].str.replace("-","")
    print(home_data["홈"])
    homeWins = home_data["홈"].str[0:2].astype(int) #int로 타입변환 // 변환안해주면 계산 안됨
    homeDraws = home_data["홈"].str[2].astype(int)
    homeLosses = home_data["홈"].str[3:5].astype(int)
    print(homeWins)
    print(homeDraws)
    print(homeLosses)

    # 승률 계산
    home_data["홈승률"] = homeWins / (homeWins + homeDraws + homeLosses )
    home_data["홈승률"] = home_data["홈승률"].round(2)  # 소수점 둘째 자리까지 반올림
    print(home_data["홈승률"])


    #######################################################(-) 제외한 방문 승률 계산
    home_data["방문"] = home_data["방문"].str.replace("-", "")
    print(home_data["방문"])
    visitWins = home_data["방문"].str[0:2].astype(int)  # int로 타입변환 // 변환안해주면 계산 안됨
    visitDraws = home_data["방문"].str[2].astype(int)
    visitLosses = home_data["방문"].str[3:5].astype(int)
    print(visitWins)
    print(visitDraws)
    print(visitLosses)

    # 승률 계산
    home_data["방문승률"] = visitWins / (visitWins + visitDraws + visitLosses )
    home_data["방문승률"] = home_data["방문승률"].round(2)  # 소수점 둘째 자리까지 반올림
    print(home_data["방문승률"])

    t통계량 , p값 = stats.ttest_ind( home_data["홈승률"] , home_data["방문승률"] )

    print(t통계량) # 0.9539726017558458

    print(p값)# 0.35273418485510766

    print( home_data['홈승률'] )
    print(home_data['방문승률'])

# hypoHome ()


#######################################################################################################################찬스 상황 일때 결정력이 좋을수록[득점권 타율(RISP)이 좋을수록] 승률이 더 높다.
def hypoRisp() :
    risp_data = pd.read_csv("crawl_csv/hitter/팀기록_타자_2024.csv", header=0, engine="python")
    risp_data = risp_data[["팀명","RISP"]]
    print(risp_data)


########################타자는 승률이 없어서 타자꺼 할때마다 가져오는 승률 구하기 코드
    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)
    win_data = win_data[["팀명", "경기", "승"]]
    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    merge_data = risp_data.merge(win_data[["팀명","승률"]], on="팀명")
    print(merge_data)

    회귀모형수식 = "승률 ~ RISP"
    선형회귀모델 = ols(회귀모형수식, data=merge_data).fit()

    print(선형회귀모델.summary())

    '''
                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.194
    Model:                            OLS   Adj. R-squared:                  0.094
    Method:                 Least Squares   F-statistic:                     1.930
    Date:                Sun, 29 Sep 2024   Prob (F-statistic):              0.202
    Time:                        16:58:17   Log-Likelihood:                 16.099
    No. Observations:                  10   AIC:                            -28.20
    Df Residuals:                       8   BIC:                            -27.59
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.1254      0.445     -0.282      0.785      -1.152       0.901
    RISP           2.1828      1.571      1.389      0.202      -1.440       5.806
    ==============================================================================
    Omnibus:                        1.397   Durbin-Watson:                   2.609
    Prob(Omnibus):                  0.497   Jarque-Bera (JB):                0.718
    Skew:                           0.063   Prob(JB):                        0.698
    Kurtosis:                       1.693   Cond. No.                         99.2
    ==============================================================================
    '''

    # 가설 검증
    if 선형회귀모델.pvalues["RISP"] < 0.05:
        print(f"찬스 상황 일때 결정력이 좋을수록[득점권 타율(RISP)이 좋을수록] 승률이 더 높다. p-value: {선형회귀모델.pvalues['RISP']:.5f}")
    else:
        print(f"찬스 상황 일때 결정력이 좋다고[득점권 타율(RISP)이 좋을수록] 승률이 더 높진 않다. p-value: {선형회귀모델.pvalues['RISP']:.5f}")
    return {"가설결과": "찬스 상황 일때 결정력이 좋다고[득점권 타율(RISP)이 좋을수록] 승률이 더 높진 않다."}

# hypoRisp()

########################################################################################################################주루플레이를 잘하는 팀일수록[도루 성공률(SB)이 좋을수록] 승률이 더 높다

def hypoSb() :
    sbData = pd.read_csv("crawl_csv/runner/팀기록_주루_2024.csv", header=0, engine="python")
    sbData = sbData[["팀명","SB%"]]
    sbData["SB"] = sbData["SB%"]

    print(sbData)

    ########################주루는 승률이 없어서 타자꺼 할때마다 가져오는 승률 구하기 코드
    # 승 수(W) 데이터 읽기
    win_data = pd.read_csv("crawl_csv/kbreport/kbreport_2024.csv", header=0, engine="python")
    # print(win_data)
    win_data = win_data[["팀명", "경기", "승"]]
    # 승률 계산
    win_data['승률'] = win_data['승'] / win_data['경기']  # 승률 계산
    win_data['승률'] = win_data['승률'].round(3)  # 소수점 반올림

    merge_data = sbData.merge(win_data[["팀명","승률"]], on="팀명")
    print(merge_data)

    회귀모형수식 = "승률 ~ SB"
    선형회귀모델 = ols(회귀모형수식, data=merge_data).fit()

    print(선형회귀모델.summary())

    '''
         OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     승률   R-squared:                       0.010
    Model:                            OLS   Adj. R-squared:                 -0.114
    Method:                 Least Squares   F-statistic:                   0.07883
    Date:                Sun, 29 Sep 2024   Prob (F-statistic):              0.786
    Time:                        17:06:31   Log-Likelihood:                 15.068
    No. Observations:                  10   AIC:                            -26.14
    Df Residuals:                       8   BIC:                            -25.53
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.5580      0.234      2.387      0.044       0.019       1.097
    SB            -0.0009      0.003     -0.281      0.786      -0.008       0.006
    ==============================================================================
    Omnibus:                        0.879   Durbin-Watson:                   1.052
    Prob(Omnibus):                  0.644   Jarque-Bera (JB):                0.564
    Skew:                           0.516   Prob(JB):                        0.754
    Kurtosis:                       2.462   Cond. No.                         921.
    ==============================================================================
    '''
    # 가설 검증
    if 선형회귀모델.pvalues["SB"] < 0.05:
        print(f"주루플레이를 잘하는 팀일수록[도루 성공률(SB)이 좋을수록] 승률이 더 높다. p-value: {선형회귀모델.pvalues['SB']:.5f}")
    else:
        print(f"주루플레이를 잘하는 팀이라도[도루 성공률(SB)이 좋을수록] 승률이 더 높진 않다 p-value: {선형회귀모델.pvalues['SB']:.5f}")
    return {"가설결과": "주루플레이를 잘하는 팀이라도[도루 성공률(SB)이 좋을수록] 승률이 더 높진 않다."}

# hypoSb()


@app.route("/hypo" , methods=["GET"] )
def hypo_js():
    results = {
        "가설1": hypoHld(),
        "가설2": hypoSac(),
        "가설3": hypoRecent(),
        "가설4": hypoEra(),
        "가설5": hypoSo(),
        "가설6": hypoPhBa(),
        "가설7": hypoNp(),
        "가설8": hypoHr(),
        "가설9": hypoRisp(),
        "가설10": hypoSb()
    }
    return jsonify(results)  # JSON 형식으로 반환

# if __name__ == '__main__':
#     app.run(debug=True)








