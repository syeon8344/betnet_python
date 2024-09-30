import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# 투수 기록 CSV에 삼진율, 볼넷삼진비율 계산해서 CSV에 포함: SO/TBF*100, TBF = IP × 2.9(병살 등 고려) + BB + H + HBP
def add_pitcher_metrics(df: pd.DataFrame):
    # IP 값("1000 1/2" 등)을 소수점으로 변환
    def convert_ip(row):
        # 공백으로 나누기
        parts = row["IP"].split(" ")
        if len(parts) == 1:
            # 공백이 없으면 그대로 반환
            return float(parts[0])
        else:
            # 정수 분수 분리
            integer_part = int(parts[0])
            fraction_part = parts[1]

            # 분수 계산
            numerator, denominator = map(int, fraction_part.split("/"))
            decimal_fraction = numerator / denominator

            # 합산값 반환
            return integer_part + decimal_fraction

    # 데이터 타입 확인 및 변환
    df["SO"] = pd.to_numeric(df["SO"], errors="coerce")
    df["IP"] = round(df.apply(convert_ip, axis="columns"), 3)
    df["BB"] = pd.to_numeric(df["BB"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df["HBP"] = pd.to_numeric(df["HBP"], errors="coerce")

    df["K%"] = round((df["SO"] / (df["IP"] * 2.9 + df["BB"] + df["H"] + df["HBP"])) * 100, 3)
    df["K/BB"] = round(df["SO"] / df["BB"], 3)
    return df


# 월간일정 CSV에 승률, 배당률, 고유코드 붙여서 저장하기
# 월간 경기 일정에 경기별 고유코드 추가
def add_match_code(df: pd.DataFrame, date: str):
    # 경기고유코드: 20240901-롯데-1400, 연월일-홈팀명-시작시간
    # apply(): 매 행마다 함수 적용, lambda 함수로 매 행마다 경기 고유코드를 생성해서 match_code 열의 내용으로 입력
    df["경기코드"] = df.apply(
        lambda row: f"{date}{row['일']}-{row['홈팀명']}-{row['시작시간'].replace(':', '')}",
        axis=1
    )

    # 경기코드 열을 DataFrame의 맨 앞에 삽입
    df.insert(0, "경기코드", df.pop("경기코드"))
    return df


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


# 매일 크롤링 작업시 회귀분석모델 생성, 월간 일정 DataFrame에 승률 및 배당금 계산 및 저장
def add_win_calc(df: pd.DataFrame):
    """
    :param df: 월간 일정 크롤링 DataFrame
    :return 각 팀별 승률과 배당금 비율 열이 포함된 DataFrame
    """
    # 다른 파일에서 호출된 경우 해당 파일 주소가 상대 위치의 시작점이므로 경로 수정 필요: app.py에서 실행시 /src
    # 현재 프로세스의 실행 경로 확인 코드
    # import os
    #
    # current_directory = os.getcwd()
    # print(f"현재 작업 디렉토리: {current_directory}")  #

    # 타자/투수/주루 특정 지표들을 독립변수로, 팀 순위를 종속변수로 해서 팀 순위를 예측해서 승패 예측에 사용
    # CSV 파일들 모으기
    hitter_csv_list = sorted(glob.glob("crawl_csv/hitter/*.csv"))
    pitcher_csv_list = sorted(glob.glob("crawl_csv/pitcher/*.csv"))
    runner_csv_list = sorted(glob.glob("crawl_csv/runner/*.csv"))
    rank_csv_list = sorted(glob.glob("crawl_csv/rank/*.csv"))
    kbreport_csv_list = sorted(glob.glob("crawl_csv/kbreport/*.csv"))
    year_df_list = []
    csv_per_year = list(zip(hitter_csv_list, pitcher_csv_list, runner_csv_list, rank_csv_list, kbreport_csv_list))

    # zip된 각 CSV 주소들을 각각 변수 이름으로 CSV 파일에서 DataFrame을 읽어와 연도별로 합친다
    """
        배당률이 변하는 문제: 매번 배당률 계산시 매일 크롤링하는 2024년 선수 데이터가 적용되어서 문제가 생긴다.
        해결방법: (일단은) 바로 전 연도까지의 데이터만 사용한다.
    """
    for hitter, pitcher, runner, rank, kbreport in csv_per_year[:-1]:  # 올해 제외 (2015~2023)
        df_hitter = pd.read_csv(hitter, encoding="utf-8")[["팀명", "RBI", "OPS"]]
        df_pitcher = pd.read_csv(pitcher, encoding="utf-8")[["팀명", "ERA", "HLD", "CG", "QS", "K%", "K/BB", "WHIP"]]
        df_runner = pd.read_csv(runner, encoding="utf-8")[["팀명", "SB%", "OOB", "PKO"]]
        df_rank = pd.read_csv(rank, encoding="utf-8")[["순위", "팀명", "승률"]]
        df_kbreport = pd.read_csv(kbreport, encoding="utf-8")[["팀명", "wOBA", "WAR"]]
        df_hitter.set_index("팀명", inplace=True)
        df_pitcher.set_index("팀명", inplace=True)
        df_runner.set_index("팀명", inplace=True)
        df_rank.set_index("팀명", inplace=True)
        df_kbreport.set_index("팀명", inplace=True)

        concat_df = pd.concat([df_rank, df_hitter, df_pitcher, df_runner, df_kbreport], axis=1)
        yearly_df = concat_df.sort_values(by="순위").reset_index()
        year_df_list.append(yearly_df)

    # 2015~2024 전체 타자/투수/주루/팀순위 DataFrame
    df_d = pd.concat(year_df_list, ignore_index=True)
    # print(df_d["팀명"].value_counts())
    # print(df_d["순위"].value_counts())
    # print(df_data.shape)
    # print(df_data.columns)

    # 표준화된 지표 추가: (열 - 열 평균)/표준편차
    def add_standardized_data(df: pd.DataFrame):
        hld_mean = df['HLD'].mean()
        hld_std_dev = df['HLD'].std()
        rbi_mean = df['RBI'].mean()
        rbi_std_dev = df['RBI'].std()
        qs_mean = df['QS'].mean()
        qs_std_dev = df['QS'].std()
        # 각 열의 표준화된 값 열 추가
        df['HLD_st'] = round((df['HLD'] - hld_mean) / hld_std_dev, 3)
        df['RBI_st'] = round((df['RBI'] - rbi_mean) / rbi_std_dev, 3)
        df['QS_st'] = round((df['QS'] - qs_mean) / qs_std_dev, 3)
        return df

    df_data = add_standardized_data(df_d)
    # 선형 회귀 모델
    # 독립 변수와 종속 변수 분리
    y = df_data["순위"].astype(int)
    x = df_data.drop(["순위", "팀명", "HLD", "RBI", "QS"], axis=1, inplace=False)  # 표준화 열의 원본 열도 제외
    # print(x.columns)
    # 훈련용, 테스트용 데이터 분리
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
    # 선형 회귀 분석 모델
    lr = LinearRegression()
    # 선형 회귀 분석 모델 훈련
    lr.fit(x_train, y_train)
    # 테스트 데이터에 대한 예측 수행: y_predict
    y_predict = lr.predict(x_test)
    # print(y_predict)
    # 훈련된 선형 회귀 분석 모델 평가
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    # print("=== 승률/배당률 선형 회귀 모델 평가 지표 ===")
    # print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}")
    # print(f"y 절편: {np.round(lr.intercept_, 2)}, 회귀계수: {np.round(lr.coef_, 2)}")
    # coef = pd.Series(data=np.round(lr.coef_, 2), index=x.columns)
    # coef.sort_values(ascending=False)
    # print(coef)
    # print("========================================")

    # 각 행에 대해 예측 수행
    # row: DataFrame의 각 행을 나타내는 매개변수
    # axis=1: 함수를 행 단위로 적용
    # df_latest = 타자/투수/주루/순위 정보가 합쳐진 작년도 DataFrame
    df_l = year_df_list[-2]
    df_latest = add_standardized_data(df_l)
    # TODO: 예측 순위가 너무 차이나서 배당률이 마이너스일 경우 + 낮은 쪽이 1.1 미만일 경우 1.1로 맞추기?
    # 9-10, 9-28 한화 SSG 경기 참고
    df["어웨이예측순위"] = df.apply(
        # 예측할 입력 데이터를 배열로 변환
        lambda row: round(lr.predict(df_latest.loc[df_latest['팀명'] == row['어웨이팀명'], x.columns])[0], 3),
        axis=1  # 행 단위로 적용
    )
    df["홈예측순위"] = df.apply(
        # 예측할 입력 데이터를 배열로 변환
        lambda row: round(lr.predict(df_latest.loc[df_latest['팀명'] == row['홈팀명'], x.columns])[0], 3),
        axis=1  # 행 단위로 적용
    )

    # 배당률 계산 및 반환 함수
    def betting_calc(home_rank_approx: float, away_rank_approx: float):
        """
        두 숫자(홈예측순위, 어웨이예측순위) 간의 거리에 따라 배당률 값을 반환.
        - 가까울수록 1.50, 1.50에 가까움
        - 멀어질수록 1.10, 1.90에 가까움
        :return 낮은 순위(강팀) 팀에 큰 확률이 등록되도록 정렬된 pd.Series
        """
        # 각 순위간의 절대값 거리
        distance = abs(home_rank_approx - away_rank_approx)

        # 거리의 최대값은 9 (1과 10의 차이)
        max_distance = 9

        # 거리 비율을 계산하여 0과 1 사이의 값으로 변환
        normalized_distance = distance / max_distance  # 0 ~ 1 범위로 정규화

        # 결과값을 0.50에서 거리 비율에 따라 조정
        result_a = round(1.50 - normalized_distance * (1.50 - 1.10), 2)  # 1.10 ~ 1.50 사이로 변환
        result_b = 3 - result_a  # result a + result b = 3
        result_high, result_low = (result_a, result_b) if result_a >= result_b else (result_b, result_a)

        # 0.50 ~ 1.00 사이의 값을 1.00 ~ 1.50 사이로 변환 예측순위가 낮은 팀이 강한 팀이므로 낮은 배당률 할당
        if home_rank_approx <= away_rank_approx:
            return pd.Series([result_low, result_high])
        else:
            return pd.Series([result_high, result_low])

    # 배당률 열 등록
    df[["어웨이배당률", "홈배당률"]] = df.apply(lambda row: betting_calc(row["어웨이예측순위"], row["홈예측순위"]), axis=1)

    # 승률 열 등록 (2-배당률 * 100)
    df["어웨이승률"] = round(2 - df["어웨이배당률"], 2)
    df["홈승률"] = round(2 - df["홈배당률"], 2)

    # 분석 시각화 1. 상관관계 히트맵
    df_view = x
    df_view['순위'] = y
    plt.figure(figsize=(12, 8))
    correlation_matrix = df_view.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # 분석 시각화 2. 산점도 행렬
    sns.pairplot(df_view, vars=df_view.columns[:5].tolist() + ['target'])  # 첫 5개 독립변수와 종속변수
    plt.show()

    return df


# 파일을 직접 열 경우 실행
if __name__ == "__main__":
    add_win_calc(pd.read_csv("../crawl_csv/monthly_schedule/월간경기일정_202409.csv", encoding="utf-8"))
