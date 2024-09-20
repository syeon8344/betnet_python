import pandas as pd

# 원본 CSV 파일을 수정해도 괜찮은가?
# -> 월간일정 CSV에 승률, 배당률, 고유코드 붙여서 저장하기?
# 경기 고유코드 추가
def add_match_code(df: pd.DataFrame):
    pass



# 삼진율 계산해서 CSV에 포함: SO/TBF*100, TBF = IP × 2.9(병살 등 고려) + BB + H + HBP
def add_strikeout_rate(df: pd.DataFrame):
    df["K%"] = (df["SO"] / (df["IP"] * 2.9 + df["BB"] + df["H"] + df["HBP"])) * 100
    return df


def add_kbb_rate(df: pd.DataFrame):
    # df["K/BB"] =
    pass
