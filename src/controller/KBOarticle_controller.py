from flask import request

from src.app import app
from src.service.KBOarticle_service import *

@app.route("/article/kbo" , methods = ['get'])
def getKboArticle():
    result = []
    getKBOnews(result)
    colsNames = ['media_company_url', 'media_company_name', 'media_company_thumb', 'url', 'title' , 'thumb']
    print(result)
    final_result = list_to_df(result , colsNames)
    get_keywords(final_result)
    return final_result

