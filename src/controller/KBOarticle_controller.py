from flask import request

from src.app import app
from src.service.KBOarticle_service import *
from src.service.article_service import *

@app.route("/article/kbo" , methods = ['get'])
def getKboArticle():

    jsonResult = main(srcText)  # [code1] 메소드실행
    print(f'jsonResult >> {jsonResult}')
    keyword_result = get_keyword(jsonResult)
    print(f'keyword_result >> {keyword_result}')
    jsonResult.append(keyword_result)
    print(jsonResult)
    print(len(jsonResult))
    return jsonResult
