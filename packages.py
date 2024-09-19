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


"""
    9.19
    특정 경기가 취소되었음을 관리자측에서 설정한다
    크롤링된 CSV -> 웹페이지 일정 목록 -> 유저 -> 일정 목록에서 경기 선택 -> 나중에 취소된 경기를 관리자가 지정 (<->DB)
    경기 DB: 경기 인덱스, 경기 상황
    
"""
