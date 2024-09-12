import collections
from collections import Counter
from operator import index
from functools import reduce
import operator



import pandas as pd
# 각 질문에 선택하는 보기에 따라 , 선택된 팀을 포함하는 데이터를 딕셔너리에 담는다.
questionnaire = {
    1: {
        'home_runs': ['삼성', '엔씨', '기아'],
        'clutch_hits': ['SSG', '키움', '두산'],
        'obp': ['키움', '엘지', '엔씨']
    },
    2: {
        '2b': ['롯데', '키움', '두산'],
        '3b': ['한화', '엘지', '기아'],
        'hr': ['삼성', '엔씨', '케이티']
    },
    3: {
        'avg': ['기아', '엘지', '롯데'],
        'slg': ['케이티', '엔씨', '삼성'],
        'balance': ['한화', '기아', 'SSG']
    },
    4: {
        'sac_sf': ['삼성', '케이티', '엔씨'],
        'power_rbi': ['기아', '엘지', '두산']
    },
    5: {
        'important': ['기아', '한화', '롯데'],
        'overall': ['케이티', '두산', 'SSG']
    },
    6: {
        'high': ['삼성', '엔씨', '기아'],
        'obp': ['케이티', '엘지', '엔씨'],
        'balance': ['한화', '키움', 'SSG']
    },
    7: {
        'aggressive': ['케이티', '기아', 'KT'],
        'strategic': ['두산', '엘지', 'SSG'],
        'flexible': ['키움', '엔씨', '한화']
    },
    8: {
        'power': ['기아', '엘지', '롯데'],
        'technical': ['SSG', 'KT', '롯데'],
        'teamwork': ['엘지', '두산', 'NC']
    },
    9: {
        'yes': ['SSG', '키움', '두산'],
        'no': ['케이티', '롯데', '엔씨']
    },
    10: {
        'yes': ['엘지', '두산', '기아'],
        'power_rbi': ['케이티', '롯데', '삼성']
    }
}

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











