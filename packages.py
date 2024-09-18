# 설치할 패키지들 목록

# 마우스로 클릭 > 빨간 전구 아이콘 > 패키지 ~~~ 설치
from flask import Flask
from flask_cors import CORS
from selenium import webdriver
import webdriver_manager
from bs4 import BeautifulSoup
import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud, STOPWORDS


# 첫 번째 데이터프레임 df1
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd']
})

# 두 번째 데이터프레임 df2
df2 = pd.DataFrame({
    'A': [1, 2, 3, 5, 4, 6],
    'B': ['a', 'b', 'c', 'e', 'd', 'f']
})

# TODO: KBO_crawl.py에서 get_monthly_schedule() 수정
# 1. 첫 번째 데이터프레임을 기준으로 중복되지 않는 행들 추출
# df2.apply(tuple, 1): df2의 각 행을 튜플로 변환합니다. 1은 행(axis=1)을 기준으로 변환함을 의미합니다. e.g. 0 (1, a)
# df1.apply(tuple, 1)는 df1의 각 행을 튜플로 변환합니다.
# df2.apply(tuple, 1).isin(df1.apply(tuple, 1))는 df2의 각 행이 df1의 어떤 행과 동일한지 여부를 확인합니다.
"""
0     True
1     True
2     True
3    False
4    False
5    False
dtype: bool
"""
# ~은 NOT 논리, 불린값을 반대로
# -> 이 표현식은 df2에서 df1에 존재하지 않는 행만을 선택합니다.
df2_unique = df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1))]

# 2. 첫 번째 데이터프레임에 중복되지 않는 행들을 추가
df_combined = pd.concat([df1, df2_unique], ignore_index=True)

# 3. 결과 데이터프레임의 인덱스를 리셋
df_combined = df_combined.reset_index(drop=True)

print("결과 데이터프레임:")
print(df_combined)

# df2에서 df1과 중복되지 않는 행 선택
df2_unique = df2.merge(df1, how='left', indicator=True)
print(df2_unique)
df2_unique = df2_unique[df2_unique['_merge'] == 'left_only'].drop(columns=['_merge'])

print(df2_unique)

# 2. 첫 번째 데이터프레임에 중복되지 않는 행들을 추가
df_combined = pd.concat([df1, df2_unique], ignore_index=True)

# 3. 결과 데이터프레임의 인덱스를 리셋
df_combined = df_combined.reset_index(drop=True)
print(df_combined)

# 두 DataFrame을 연결하고 중복된 행 제거
combined = pd.concat([df1, df2]).drop_duplicates(keep=False)

print("중복이 아닌 행들:")
print(combined)
res = pd.concat([df1, combined], ignore_index=True)
print(res)