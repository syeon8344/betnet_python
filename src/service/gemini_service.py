
from src.app import app
from flask import Flask, request, jsonify

import google.generativeai as genai
# 터미널에 pip install -U google-generativeai
def geminiService(keyword):
    # API를 자신의 API 키로 설정합니다.
    print( keyword )
    genai.configure(api_key='AIzaSyClbOBrF4jWqdwug_D9Xbd21R2HXNPrxNY')
    # gemini 설정
    generation_config = {
        "temperature": 0, # 정확도
    }

    print(generation_config)
    # 생성 모델을 초기화합니다 (예: gemini-pro)
    model = genai.GenerativeModel('gemini-pro')
    print( model )
    # 빈 채팅 기록으로 새 채팅 세션을 시작합니다.
    chat = model.start_chat(history=[])
    # 사용자의 메시지를 챗봇 모델에 전송합니다.
    print( keyword )
    response = chat.send_message(keyword)
    # 챗봇의 응답을 반환합니다.
    print(response.text)
    return response.text

