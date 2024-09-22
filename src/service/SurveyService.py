import collections
import glob
from collections import Counter
from operator import index
from functools import reduce
import operator
import pandas as pd

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

import pandas as pd

# 1. 홈런율이 있는 데이터를 가져오기
hitter_csv_list = glob.glob("./crawl_csv/hitter/*.csv")
pitcher_csv_list = glob.glob("./crawl_csv/pitcher/*.csv")
runner_csv_list = glob.glob("./crawl_csv/runner/*.csv")
rank_csv_list = glob.glob("./crawl_csv/rank/*.csv")

csv_per_year = zip(hitter_csv_list, pitcher_csv_list, runner_csv_list, rank_csv_list)
# print( csv_per_year )
for hitter, pitcher, runner, rank in csv_per_year:
    # print(hitter, pitcher, runner, rank)
    df_hitter = pd.read_csv(hitter, encoding="utf-8")[["팀명", "HR" , "SF" , "SLG" , "OBP" , "2B" , "3B" , "AVG" , "OPS"  , "R" ,
                                                       "TB" , "RBI" , "PH-BA" , "RISP" ]]

# print( df_hitter )
# print( df_hitter.sort_values('RISP'))
# print( df_hitter.sort_values('RISP').tail(3)['팀명'].to_list() )
questionnaire = {
    1: {
        'home_runs': df_hitter.sort_values('HR').tail(3)['팀명'].to_list(),    # HR 홈런율
        'clutch_hits': df_hitter.sort_values('SLG').tail(3)['팀명'].to_list(),  # SLG 장타율
        'obp': df_hitter.sort_values('OBP').tail(3)['팀명'].to_list()           # OBP 출루율
    },
    2: {
        '2b': df_hitter.sort_values('2B').tail(3)['팀명'].to_list(),       # 2B 2루타
        '3b': df_hitter.sort_values('3B').tail(3)['팀명'].to_list(),       # 3B 3루타
        'hr': df_hitter.sort_values('HR').tail(3)['팀명'].to_list()       # HR 홈런율
    },
    3: {
        'avg': df_hitter.sort_values('AVG').tail(3)['팀명'].to_list(),      # AVG 타율
        'slg': df_hitter.sort_values('SLG').tail(3)['팀명'].to_list(),     # SLG 장타율
        'balance': df_hitter.sort_values('OPS').tail(3)['팀명'].to_list()      # OPS 타율 + 출루율
    },
    4: {
        'sac_sf': df_hitter.sort_values('SF').tail(3)['팀명'].to_list(),  # SF 희생타
        'power_rbi': df_hitter.sort_values('RISP').tail(3)['팀명'].to_list() # RISP 득점권타율
    },
    5: {
        'important': df_hitter.sort_values('SF').tail(3)['팀명'].to_list(),    # SF 희생 플레이
        'overall': df_hitter.sort_values('R').tail(3)['팀명'].to_list()     # R 득점
    },
    6: {
        'high': df_hitter.sort_values('HR').tail(3)['팀명'].to_list(),         # HR 홈런율
        'obp': df_hitter.sort_values('OBP').tail(3)['팀명'].to_list(),         # OBP 출루율
        'balance': df_hitter.sort_values('TB').tail(3)['팀명'].to_list()          # TB 총루타
    },
    7: {
        'aggressive': df_hitter.sort_values('RBI').tail(3)['팀명'].to_list(),  # RBI: 타점,
        'strategic': df_hitter.sort_values('OPS').tail(3)['팀명'].to_list(),   # OPS: 출루율 + 장타율
        'flexible': df_hitter.sort_values('PH-BA').tail(3)['팀명'].to_list()     # PH-BA: 대타타율
    },
    8: {
        'power': df_hitter.sort_values('RISP').tail(3)['팀명'].to_list(),       # RISP: 득점권타율
        'technical': df_hitter.sort_values('OPS').tail(3)['팀명'].to_list(),   # OPS: 출루율 + 장타율
        'teamwork': df_hitter.sort_values('OBP').tail(3)['팀명'].to_list()      # OBP: 출루율
    },
    9: {
        'yes': df_hitter.sort_values('RBI').tail(3)['팀명'].to_list(),     #  RBI: 타점,
        'no': df_hitter.sort_values('AVG').tail(3)['팀명'].to_list()       #  AVG: 타율
    },
    10: {
        'yes': df_hitter.sort_values('SF').tail(3)['팀명'].to_list(),          # SF: 희생플라이
        'power_rbi': df_hitter.sort_values('SLG').tail(3)['팀명'].to_list()    #  SLG: 장타율
    }
}
# print(questionnaire)
# 각 질문에 선택하는 보기에 따라 , 선택된 팀을 포함하는 데이터를 딕셔너리에 담는다.

# 사용자의 선택을 입력받는 함수
def recommend(question_number, choice):
    # recommendations 딕셔너리에서 질문 번호가 유효한지 확인
    if question_number in questionnaire:
        # 해당 질문 번호에서 사용자가 선택한 옵션이 유효한지 확인
        if choice in questionnaire[question_number]:
            # 유효한 선택인 경우, 해당 선택에 대한 추천 팀 목록 반환
            return questionnaire[question_number][choice]
    # 만약 질문 번호 또는 선택이 유효하지 않다면 빈 리스트 반환
    return []
def recommendTest( param ) :
    print(param)
    index = 1;
    resultList = []
    for key in param :
        # 현재 질문 번호와 선택된 옵션을 출력
        print( param[ key ] )
        # 사용자의 선택에 따라 추천된 팀 목록을 가져옴
        result  = recommend( index , param[key] )
        # 추천된 팀 목록을 결과 리스트에 추가
        resultList.append( result )
        # 다음 질문으로 이동하기 위해 인덱스 1씩 증가
        index +=1
    print( resultList )


    list2 = list(reduce(operator.add, resultList ))
    print(f'list : {list2}')
    cnt = Counter(list2).most_common(1)
    print(cnt)

    team = cnt[0][0]
    print(team)

    return team











