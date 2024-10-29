
from src.app import app
from flask import Flask, request, jsonify

import google.generativeai as genai
#from google.generativeai.types import HarmCategory, HarmBlockThreshold
# 터미널에 pip install -U google-generativeai
"""
python 3.8 버전으로 gemini API 사용하기
alt + F12 또는 왼쪽 아래 터미널 아이콘에서
pip install -U google-generativeai
이후 grpcio-status 1.62.3 requires protobuf>=4.21.6, but you have protobuf 3.20.3 which is incompatible. 같은 오류 발생시 아래 코드 실행
pip install protobuf==3.20.*
이후 패키지에서 generativeai-for-python3.8 설치
protobuf 오류가 또 발생하면 위에 pip install protobuf... 코드 다시 실행
"""

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
    response = chat.send_message(keyword,
                                 # safety_settings={
                                 #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                 #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                 #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                 #     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                                 #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                 # }
                                 )
    # 챗봇의 응답을 반환합니다.
    print(response.text)
    return response.text

