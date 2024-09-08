# 설치할 패키지들 목록
# 마우스로 클릭 > 빨간 전구 아이콘 > 패키지 ~~~ 설치
from flask import Flask
from selenium import webdriver
import webdriver_manager
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS