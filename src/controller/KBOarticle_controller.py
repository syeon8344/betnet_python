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
    return final_result

@app.route("/article/team" , methods = ['get'])
def getTeamArticle():
    result = []
    srcText = "기아"
    get_team_article(result , srcText)
    colsNames = ['media_company_url', 'media_company_name', 'media_company_thumb', 'url', 'title' , 'thumb']
    print(result)
    final_result = list_to_df(result , colsNames)
    return final_result