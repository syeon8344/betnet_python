from flask import request

from src.app import app
from src.service.article_service import *

# article 페이지에 띄울 기사
@app.route("/article/kbo" , methods = ['get'])
def getArticle():
    srcText = ['KIA야구' , '삼성야구' , 'LG야구' , '두산야구' , 'KT야구' , '한화야구' , 'SSG야구' , '롯데야구' , 'NC야구' , '키움야구']
    jsonResult = main(srcText)  # [code1] 메소드실행
    print(f'jsonResult >> {jsonResult}')
    return jsonResult

# 텍스트 빈도분석
@app.route("/article/text" , methods = ['get'])
def forText():
    srcText = ['KIA야구' , '삼성야구' , 'LG야구' , '두산야구' , 'KT야구' , '한화야구' , 'SSG야구' , '롯데야구' , 'NC야구' , '키움야구']
    jsonResult = for_text(srcText)  # [code1] 메소드실행
    print(f'jsonResult >> {jsonResult}')
    keyword_result = get_keyword(jsonResult , srcText)
    print(f'keyword_result >> {keyword_result}')
    return keyword_result

# 메인페이지에 띄울 kbo 기사
@app.route("/article/main" , methods = ['get'])
def getKboArticle():
    srcText = ['KBO']
    jsonResult = get_kbo_article(srcText)  # [code1] 메소드실행
    print(f'jsonResult >> {jsonResult}')
    return jsonResult
