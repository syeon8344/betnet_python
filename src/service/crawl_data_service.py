import pandas as pd


# 월간일정 CSV에 승률, 배당률, 고유코드 붙여서 저장하기
# 월간 경기 일정에 경기별 고유코드 추가
def add_match_code(df: pd.DataFrame, date: str):
    # 경기고유코드: 20240901-롯데-1400, 연월일-홈팀명-시작시간
    # apply(): 매 행마다 함수 적용, lambda 함수로 매 행마다 경기 고유코드를 생성해서 match_code 열의 내용으로 입력
    df["경기코드"] = df.apply(
        lambda row: f"{date}{row['일']}-{row['홈팀명']}-{row['시작시간'].replace(':', '')}",
        axis=1
    )
    return df


# 투수 기록 CSV에 삼진율, 볼넷삼진비율 계산해서 CSV에 포함: SO/TBF*100, TBF = IP × 2.9(병살 등 고려) + BB + H + HBP
def add_pitcher_metrics(df: pd.DataFrame):
    # 데이터 타입 확인 및 변환
    df["SO"] = pd.to_numeric(df["SO"], errors='coerce')
    df["IP"] = round(df.apply(convert_ip, axis='columns'), 3)
    df["BB"] = pd.to_numeric(df["BB"], errors='coerce')
    df["H"] = pd.to_numeric(df["H"], errors='coerce')
    df["HBP"] = pd.to_numeric(df["HBP"], errors='coerce')

    df["K%"] = round((df["SO"] / (df["IP"] * 2.9 + df["BB"] + df["H"] + df["HBP"])) * 100, 3)
    df["K/BB"] = round(df["SO"] / df["BB"], 3)
    return df


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
