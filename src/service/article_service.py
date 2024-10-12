import json
import urllib.parse
import urllib.request

import pandas as pd

clientId = "vkFgWXdbZfUneulMiPBq"
clientsecret = "HlKBE2kztf"

def getRequestUrl(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id" , clientId)
    req.add_header("X-Naver-Client-Secret" , clientsecret)
    try:
        res = urllib.request.urlopen(req)
        # print(f'>>code2 요청URL 결과상태 : {res.getcode()}')
        if res.getcode() == 200:
            # print(res.read().decode('utf-8'))
            return res.read().decode('utf-8')
    except Exception as e:
        print(e)
        return None

def getNaverSearch(node , src , page_start , display):
    base = "https://openapi.naver.com/v1/search"    # 1. 요청 url의 기본 주소
    node = f'/{node}.json'  # 2. 요청 url의 검색 대상과 json 파일 이름
        # https://openapi.naver.com/v1/search/news.json 네이버 공문 검색 요청 url
    parameters = f'?query={urllib.parse.quote(src)}&start={page_start}&display={display}'   # 3. 요청 url의 파라미터
        # https://openapi.naver.com/v1/search/news.xml?query=검색어&display=한번표기할개수&start=시작번호
    url = base + node + parameters  # 4. url 합치기
    # print(f'>>code3 요청URL : {url}') # 확인
    responseDecode = getRequestUrl(url)   # 5. url 요청을 하고 응답객체 받기 , [code2]

    if responseDecode == None: return None    # 6. 만약에 url 객체가 없으면 None 반환
    else: return json.loads(responseDecode) # 7. 응답객체가 있으면 JSON 형식으로 변환
        # json.loads(문자열) : JSON 형식으로 변환

def getPostData(post , jsonResult , cnt , src):
    # 응답받은 객체의 요소들 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#응답
    title = post['title']   # 뉴스 기사의 제목
    description = post['description']   # 뉴스 기사의 내용을 요약한 패시지 정보.
    org_link = post['originallink']     # 뉴스 기사 원문의 URL
    pubDate = post['pubDate']

    # 딕셔너리 생성
    dic = {
        'cnt': cnt, 'title': title, 'description': description, 'org_link': org_link, 'pubDate': pubDate
    }

    # src를 기준으로 jsonResult에 저장
    if src not in jsonResult:
        jsonResult[src] = []  # 새로운 리스트 생성
    jsonResult[src].append(dic)  # 해당 src에 딕셔너리 추가

def get_keyword(jsonResult, srcText):
    import re  # 정규표현식 모듈
    from konlpy.tag import Okt
    from collections import Counter
    stopwords = {'마지막', '시리즈', '프로야구', '스폰서', '코스피', '유플러스', '하반기',
                 '글로벌', '신입사원', '연구원', '부회장', '먹거리', '대통령', '플라스틱', '결정전', '신기록',
                 '굿바이', '라이프', '끝내기', '생중계', '마침표'}
    result = []
    okt = Okt()  # 품사 태깅 객체를 한 번만 생성
    for src in srcText:
        print(f'Processing src: {src}')
        description = ''
        # jsonResult가 딕셔너리이므로 src에 해당하는 리스트에 접근
        if src in jsonResult:
            articles = jsonResult[src]
            for i in articles:
                if 'description' in i:
                    # 전처리(정규표현식) / (특수문자 제거)
                    description += re.sub(r'[^\w]', ' ', i['description']) + ' '
        # 명사 추출
        tag_words = okt.nouns(description)
        # 불용어 제거 및 3글자 이상 필터링
        filtered_words = [
            word for word in tag_words
            if word not in stopwords and len(word) >= 3
        ]
        # 데이터 분석 - 단어 빈도 분석
        wordsCount = Counter(filtered_words)
        # 단어 빈도 (Counter) 객체를 딕셔너리화
        word_count = {}
        for tag, count in wordsCount.most_common(50):
            if len(tag) >= 3:  # 단어의 길이가 3글자 이상인 경우
                word_count[tag] = count
                if len(word_count) >= 5:  # 5개 이상이면 종료
                    break
        # src 값을 포함하여 결과에 추가
        result.append({'src': src, 'keywords': word_count})
    return result

# 데이터 분석을 위한 300개 크롤링
def for_text(srcText):
    node = 'news'   # 1. 크롤링할 대상 [ 네이버 제공하는 검색대상 : 1. news 2. blog 3. shop 등등 ] - 공문 참고 : https://developers.naver.com/docs/serviceapi/search/blog/blog.md#%EB%B8%94%EB%A1%9C%EA%B7%B8
    # srcText = 'KBO'   # 2. 사용자 입력으로 받은 검색어 변수
    cnt = 0 # 3. 검색 결과 개수
    jsonResult = {} # 4. 검색 결과를 정리하여 저장할 리스트 변수
    total_results_needed = 100  # 원하는 결과 개수

    for src in srcText:
        print(f'src >> {src}')
        total_fetched = 0  # 현재까지 가져온 결과 개수
        start = 1  # 시작 인덱스
        while total_fetched < total_results_needed:
            # 5. 1부터 100까지 검색 결과를 처리한다. # 네이버 뉴스 결과에 대한 응답을 저장하는 객체
            jsonResponse = getNaverSearch(node , src , start , 100) # 5. [code 3]
                # jsonResponse[ total : 검색결과개수 , start : 검색시작위치 , display : 한 번에 표시할 검색 결과 개수 , item : 개별 검색 결과]
            # print(f'>> jsonResponse : {jsonResponse}')
            total = jsonResponse['total']   # 6. 전체 검색 결과 개수
            print(total)
            # 7. 응답객체가 None이 아니면서 응답객체의 display 가 0이 아니면 무한 반복 , url 응답객체가 없을때까지
            while ((jsonResponse != None) and (jsonResponse['display'] != 0)):

                # 8. 검색결과 리스트(items)에서 하나씩 item(post) 호출 # 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4
                for post in jsonResponse['items']:  # 응답받은 검색 결과 중에서 한 개를 저장한 객체
                    cnt += 1    # 응답 개수 1 증가
                    # 9. [code 3] 검색 결과 한개를 처리한다.
                    getPostData(post , jsonResult , cnt , src)
                total_fetched += jsonResponse['display']
                print(f'total_fetched >> {total_fetched}')
                # 300개씩 가져오기
                if total_results_needed <= total_fetched:
                    break
                # 10. start를 display 만큼 증가시킨다.
                start = jsonResponse['start'] + jsonResponse['display']
                # 11. 첫번째요청 1 , 100, 두번째요청 101 , 100 세번째요청 201 , 100
                jsonResponse = getNaverSearch(node , src , start , 100)
                # print(jsonResult)
                # print(len(jsonResult))

    return jsonResult

# article 페이지에 띄울 기사
def main(srcText):
    node = 'news'   # 1. 크롤링할 대상 [ 네이버 제공하는 검색대상 : 1. news 2. blog 3. shop 등등 ] - 공문 참고 : https://developers.naver.com/docs/serviceapi/search/blog/blog.md#%EB%B8%94%EB%A1%9C%EA%B7%B8
    # srcText = 'KBO'   # 2. 사용자 입력으로 받은 검색어 변수
    cnt = 0 # 3. 검색 결과 개수
    jsonResult = {} # 4. 검색 결과를 정리하여 저장할 리스트 변수
    total_results_needed = 10  # 원하는 결과 개수


    for src in srcText:
        print(f'src >> {src}')
        total_fetched = 0  # 현재까지 가져온 결과 개수
        start = 1  # 시작 인덱스
        while total_fetched < total_results_needed:
            # 5. 1부터 100까지 검색 결과를 처리한다. # 네이버 뉴스 결과에 대한 응답을 저장하는 객체
            jsonResponse = getNaverSearch(node , src , start , 10) # 5. [code 3]
                # jsonResponse[ total : 검색결과개수 , start : 검색시작위치 , display : 한 번에 표시할 검색 결과 개수 , item : 개별 검색 결과]
            # print(f'>> jsonResponse : {jsonResponse}')
            total = jsonResponse['total']   # 6. 전체 검색 결과 개수
            print(total)
            # 7. 응답객체가 None이 아니면서 응답객체의 display 가 0이 아니면 무한 반복 , url 응답객체가 없을때까지
            while ((jsonResponse != None) and (jsonResponse['display'] != 0)):

                # 8. 검색결과 리스트(items)에서 하나씩 item(post) 호출 # 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4
                for post in jsonResponse['items']:  # 응답받은 검색 결과 중에서 한 개를 저장한 객체
                    cnt += 1    # 응답 개수 1 증가
                    # 9. [code 3] 검색 결과 한개를 처리한다.
                    getPostData(post , jsonResult , cnt , src)
                total_fetched += jsonResponse['display']
                print(f'total_fetched >> {total_fetched}')
                # 300개씩 가져오기
                if total_results_needed <= total_fetched:
                    break
                # 10. start를 display 만큼 증가시킨다.
                start = jsonResponse['start'] + jsonResponse['display']
                # 11. 첫번째요청 1 , 100, 두번째요청 101 , 100 세번째요청 201 , 100
                jsonResponse = getNaverSearch(node , src , start , 100)
                # print(jsonResult)
                # print(len(jsonResult))

    return jsonResult

# 메인페이지에 띄울 기사
def get_kbo_article(srcText):
    node = 'news'  # 1. 크롤링할 대상 [ 네이버 제공하는 검색대상 : 1. news 2. blog 3. shop 등등 ] - 공문 참고 : https://developers.naver.com/docs/serviceapi/search/blog/blog.md#%EB%B8%94%EB%A1%9C%EA%B7%B8
    # srcText = 'KBO'   # 2. 사용자 입력으로 받은 검색어 변수
    cnt = 0  # 3. 검색 결과 개수
    jsonResult = {}  # 4. 검색 결과를 정리하여 저장할 리스트 변수
    total_results_needed = 5  # 원하는 결과 개수

    for src in srcText:
        print(f'src >> {src}')
        total_fetched = 0  # 현재까지 가져온 결과 개수
        start = 1  # 시작 인덱스
        while total_fetched < total_results_needed:
            # 5. 1부터 100까지 검색 결과를 처리한다. # 네이버 뉴스 결과에 대한 응답을 저장하는 객체
            jsonResponse = getNaverSearch(node, src, start, 5)  # 5. [code 3]
            # jsonResponse[ total : 검색결과개수 , start : 검색시작위치 , display : 한 번에 표시할 검색 결과 개수 , item : 개별 검색 결과]
            # print(f'>> jsonResponse : {jsonResponse}')
            total = jsonResponse['total']  # 6. 전체 검색 결과 개수
            print(total)
            # 7. 응답객체가 None이 아니면서 응답객체의 display 가 0이 아니면 무한 반복 , url 응답객체가 없을때까지
            while ((jsonResponse != None) and (jsonResponse['display'] != 0)):

                # 8. 검색결과 리스트(items)에서 하나씩 item(post) 호출 # 공문 : https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4
                for post in jsonResponse['items']:  # 응답받은 검색 결과 중에서 한 개를 저장한 객체
                    cnt += 1  # 응답 개수 1 증가
                    # 9. [code 3] 검색 결과 한개를 처리한다.
                    getPostData(post, jsonResult, cnt, src)
                total_fetched += jsonResponse['display']
                print(f'total_fetched >> {total_fetched}')
                # 300개씩 가져오기
                if total_results_needed <= total_fetched:
                    break
                # 10. start를 display 만큼 증가시킨다.
                start = jsonResponse['start'] + jsonResponse['display']
                # 11. 첫번째요청 1 , 100, 두번째요청 101 , 100 세번째요청 201 , 100
                jsonResponse = getNaverSearch(node, src, start, 100)
                # print(jsonResult)
                # print(len(jsonResult))
    return jsonResult


# if __name__ == "__main__":
#     srcText = ['KBO']
#     jsonResult = get_kbo_article(srcText)  # [code1] 메소드실행
#     print(f'jsonResult >> {jsonResult}')
