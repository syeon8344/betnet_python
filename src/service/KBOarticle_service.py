'''
팀별/선수별 타이틀 / 본문 20~30개 텍스트 분석
선수는 검색 검색 순위는  5위까지 csv 파일 처리

KBO 메인페이지 5~10개

썸네일 타이틀 링크 언론사

타이틀 안에 링크 -> 클릭하면 본문으로 이동
'''

# 1. 모듈 가져오기
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import json
import time
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

# webdriver 객체 생성
options = Options()  # 웹드라이버 설정

options.add_argument("--headless")  # 브라우저 GUI를 표시하지 않음
options.add_argument("--no-sandbox")  # 보안 샌드박스 비활성화
options.add_argument("--disable-gpu")
options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    )

wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
# 메인페이지 kbo 관련 기사
def getKBOnews(result):
    for i in range(1 ,36):
        wd.get(f"https://search.daum.net/search?nil_suggest=btn&w=news&DA=PGD&cluster=y&q=kbo&p={i}")
        time.sleep(1)
        try:
            html = wd.page_source
            soupCB1 = BeautifulSoup(html, "html.parser")
            # print(soupCB1.prettify())
            # 뉴스 리스트
            news_list = soupCB1.select_one(".c-list-basic")
            # print(news_list)
            for row in news_list.select("li"):
                div = row.select("div");
                # print(len(div))
                if len(div) <= 3 :  # 만약에 열이 개수가 0개이면 div 깨짐
                    continue    # 가장 가까운 반복문으로 이동 , 아래 코드는 실행되지 않는다.
                content_box = row.select_one(".c-item-content")
                media_company_box = row.select_one(".c-tit-doc")
                media_company_url = media_company_box.select_one(".item-writer")["href"]
                # print(f'언론사 링크 : {media_company_url}')
                media_company_name = media_company_box.select_one(".tit_item > span").string
                # print(f'언론사 이름 : {media_company_name}')
                media_company_thumb = media_company_box.select_one(".item-writer img")["src"]
                if "data:image" in media_company_thumb:
                    media_company_thumb = media_company_box.select_one(".item-writer img").get("data-original-src")
                # print(f'언론사 썸네일 : {media_company_thumb}')
                url = content_box.select_one(".item-title a")["href"]
                # print(f'기사 링크 : {url}')
                title = content_box.select_one(".item-title a").text.strip()
                # print(f'기사 제목 : {title}')
                thumb = content_box.select_one(".item-thumb img")["src"]
                if "data:image" in thumb:
                    thumb = content_box.select_one(".item-thumb img").get("data-original-src")
                # print(f'기사 썸네일 : {thumb}')
                # 리스트에 담기
                news = [media_company_url, media_company_name, media_company_thumb, url, title , thumb];  # print(news)
                result.append(news)
        except Exception as e:
            print(e)

    return result

def get_keywords(result):
    import re  # 정규표현식 모듈

    title = ''
    for item in result:
        # 만약에 item(요소) 내 'message' 라는 key가 존재하면
        if 'title' in item.keys():
            # print(item)
            # print(item['message'])  # 분석할 문장
            # 전처리(정규표현식) / (특수문자 제거)
            # 분석할 문장내 정규표현식을 이용한 특수문자를 제거하고 공백으로 치환한다.
            title = title + re.sub(r'[^\w]', ' ', item['title']) + ''
            # print(message)
    # print(message)  # 확인

    # 1-3 품사 태깅 : 명사 추출
    from konlpy.tag import Okt
    okt = Okt()  # 품사 태깅 객체 생성
    tag_words = okt.nouns(title)  # 품사(명사) 태깅하기
    # print(tag_words)    # 확인

    # 2. 데이터 분석
    # 2-1 단어빈도분석
    from collections import Counter
    wordsCount = Counter(tag_words)
    # print(wordsCount)
    # Counter(리스트 , 튜플 , 문자열) # 요소들의 빈도 수 계산 객체

    # 2-2 단어 빈도 (Counter) 객체를 딕셔너리화
    word_count = {}  # 빈 딕셔너리 생성
    # word_count = dict() vs word_count = {}
    for tag, count in wordsCount.most_common(80):
        if len(tag) > 2:  # 단어의 길이가 1글자인 경우 제외
            word_count[tag] = count
            # {단어 : 빈도수}
    print(word_count)   # 확인

# 팀별 뉴스 기사 100개 이상
def get_team_article(result , srcText):
    for i in range(1 ,46):
        wd.get(f"https://search.daum.net/search?nil_suggest=btn&w=news&DA=PGD&cluster=y&q={srcText}+야구&p={i}")
        time.sleep(1)
        try: # urllib.parse.quote(srcText)
            html = wd.page_source
            soupCB1 = BeautifulSoup(html, "html.parser")
            # print(soupCB1.prettify())
            # 뉴스 리스트
            news_list = soupCB1.select_one(".c-list-basic")
            # print(news_list)
            for row in news_list.select("li"):
                div = row.select(".c-item-content > div");
                # print(len(div))
                if len(div) <=0 :  # 만약에 열이 개수가 0개이면 div 깨짐
                    continue    # 가장 가까운 반복문으로 이동 , 아래 코드는 실행되지 않는다.
                content_box = row.select_one(".c-item-content")
                media_company_box = row.select_one(".c-tit-doc")
                media_company_url = media_company_box.select_one(".item-writer")["href"]
                # print(f'언론사 링크 : {media_company_url}')
                media_company_name = media_company_box.select_one(".tit_item > span").string
                # print(f'언론사 이름 : {media_company_name}')
                media_company_thumb = media_company_box.select_one(".item-writer img")["src"]
                if "data:image" in media_company_thumb:
                    media_company_thumb = media_company_box.select_one(".item-writer img").get("data-original-src")
                # print(f'언론사 썸네일 : {media_company_thumb}')
                url = content_box.select_one(".item-title a")["href"]
                # print(f'기사 링크 : {url}')
                title = content_box.select_one(".item-title a").text.strip()
                # print(f'기사 제목 : {title}')
                thumb = content_box.select_one(".item-thumb img")["src"]
                if "data:image" in thumb:
                    thumb = content_box.select_one(".item-thumb img").get("data-original-src")
                # print(f'기사 썸네일 : {thumb}')
                # 리스트에 담기
                news = [media_company_url, media_company_name, media_company_thumb, url, title , thumb];  # print(news)
                result.append(news)
        except Exception as e:
            print(e)

    return result


def list_to_df(result, colsNames):
    df = pd.DataFrame(result , columns=colsNames)
    jsonResult = df.to_json(orient='records' , force_ascii=False)
    # print(jsonResult)
    result = json.loads(jsonResult)    # json.loads() 문자열타입(json형식) ---> py타입(json형식) 변환
    return result

if __name__ == "__main__":
    result = []
    srcText = "기아"
    getKBOnews(result)
    colsNames = ['media_company_url', 'media_company_name', 'media_company_thumb', 'url', 'title', 'thumb']
    print(result)
    print(len(result))
    result2 = list_to_df(result, colsNames)
    get_keywords(result2)

