# 2_미니챗봇.py
from typing import TypeVar

from flask import request, abort, jsonify
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import requests
import csv

from src.service.salary_service import predictSalary
import src.ball_gpt.seq2seq.seq2seqfromclass as s2s

# RNN 기본구조 : 1. 데이터수집 2.전처리 3.토큰화/패딩 4. 모델구축 5.모델학습 6.모델평가(튜닝) 7.모델예측

# CSV 파일에서 선수 이름을 로드하는 함수
def load_player_names(filename='crawl_csv/stat2024.csv'):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        player_names = [row[11] for row in reader]  # 첫 번째 열에 선수 이름이 있다고 가정
    return player_names


# 선수 이름 리스트 로드
player_names = load_player_names()


def schedule():
    # 쿼리 문자열에서 year와 month 가져오기
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    if year and month:
        date = f'{year:04d}{month:02d}'
    else:
        date = pd.to_datetime('today').strftime('%Y%m')  # 날짜 입력이 없을 시 현재 년도 월

    try:
        # 필요한 열만 불러오기
        columns_to_load = ['연도', '월', '일', '시작시간', '홈팀명', '어웨이팀명']
        df = pd.read_csv(f'crawl_csv/monthly_schedule/월간경기일정_{date}.csv', encoding='utf-8', dtype=str,
                         usecols=columns_to_load)
    except FileNotFoundError:
        print('/monthlyschedule: 해당 연월의 월간경기일정 파일이 없습니다.')
        return abort(404)  # 404 Not Found 응답 반환

    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    print(json.loads(df.to_json(orient='records', force_ascii=False)))
    return jsonify(json.loads(df.to_json(orient='records', force_ascii=False)))


def salary(user_input):  # 매개변수는 전처리된 text가아니라 js에서 전달받은 user_input 전달해야함
    print('salary')
    # 입력된 질문에서 공백 기준으로 분리
    tokens = user_input.split(" ")

    for token in tokens:
        if token in player_names:
            response = predictSalary(token)
            print("찾았다.")
            print(response)
            player_info = response[0]
            예상연봉 = player_info.get('예상연봉')
            선수명 = player_info.get('선수명')
            str = f'{선수명}의 예상 연봉은 {예상연봉}입니다.'
            return str
        else:
            return {"error": f"{token} 선수는 목록에 없습니다."}

    else:
        return {"error": "선수 이름이 입력되지 않았습니다."}


def redirect_home(user_input):
    return 'http://localhost:8080/'


# 예측한 확률의 질문과 함수 매칭 딕셔너리
response_functions = {
    1: salary,
    4: redirect_home
    # 3 : 게시판 글쓰기
}

# 1. 데이터 수집 # csv , db , 함수(코드/메모리)
data = pd.read_csv("service/챗봇데이터.csv")
# print( data )

# 2. 데이터 전처리
inputs = list(data['Q'])  # 질문
outputs = list(data['A'])  # 응답

from konlpy.tag import Okt
import re  # 정규표현식

okt = Okt()


def preprocess(text):
    # 정규표현식 수정: 영어 알파벳 포함
    # result = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]', '', text)
    result = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', text)
    # 형태소 분석
    result = okt.pos(result)
    # 명사(Noun), 동사(Verb), 형용사(Adjective) 선택 가능
    result = [word for word, pos in result if pos in ['Noun', 'Verb', 'Adjective']]

    # 불용어 처리 (예시)
    stop_words = ['이', '그', '저', '것', '하다']  # 필요에 따라 수정
    result = [word for word in result if word not in stop_words]

    # 최종 반환
    return " ".join(result).strip()


# 전처리 실행  # 모든 질문을 전처리 해서 새로운 리스트
processed_inputs = [preprocess(질문) for 질문 in inputs]
# print( processed_inputs )

# 3. 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')  # 변수명=클래스명()
tokenizer.fit_on_texts(processed_inputs)  # 전처리된 단어 목록을 단어사전 생성
# print( tokenizer.word_index ) # 사전확인

input_sequences = tokenizer.texts_to_sequences(processed_inputs)  # 벡터화
# print( input_sequences )

max_sequence_length = max(len(문장) for 문장 in input_sequences)  # 여러 문장중에 가장 긴 단어의 개수
# print( max_sequence_length ) # '좋은 책 추천 해 주세요' # 5

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length)  # 패딩화 # 가장 길이가 긴 문장 기준으로 0으로 채우기
# print( input_sequences ) #  '오늘 날씨 어때요' --> [ 2  3  4 ] --> [ 0 0 2 3 4 ] # 좋은 성능을 만들기 위해 차원을 통일

# 종속변수 # 데이터프레임 --> 일반 배열 변환
# output_sequences = np.array(  outputs  )
# print( output_sequences )
output_sequences = np.array(range(len(outputs)))
# print( output_sequences )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

# 모델
model = Sequential()
print(tokenizer.word_index)
model.add(Embedding(input_dim=len(tokenizer.word_index), output_dim=50, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# 추가 LSTM 레이어
model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(
    0.01))))  # return_sequences=True를 통해 다음 LSTM 레이어에 시퀀스 전달
model.add(BatchNormalization())
model.add(Dropout(0.3))

# 또 다른 LSTM 레이어 추가
model.add(Bidirectional(LSTM(128)))  # 마지막 LSTM 레이어에서는 return_sequences=False
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(outputs), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))


# Learning Rate Scheduler 함수
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * tf.math.exp(-0.1)  # 학습률 감소
    return lr


# 2. 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])  # 학습률 감소

# 3. 데이터셋 분리
input_train, input_val, output_train, output_val = train_test_split(input_sequences, output_sequences, test_size=0.2)

# 체크포인트 및 조기 중단 설정
checkpoint_path = 'best_performed_model.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor='loss', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=5)

# 학습
batch_size = 32  # 원하는 배치 크기로 설정
history = model.fit(input_train, output_train, validation_data=(input_val, output_val),
                    callbacks=[checkpoint, early_stop],
                    epochs=30,
                    batch_size=batch_size)  # 배치 크기 지정

model_seq2 = s2s.Encoder()




# 4. 예측하기
def response(user_input):
    text = preprocess(user_input)  # 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences([text])  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences(text, maxlen=max_sequence_length)
    result = model.predict(text)  # 3. 예측
    max_index = np.argmax(result)  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    msg = outputs[max_index]  # max_index : 예측한 질문의 위치 . # msg : 예윽한 질문의 위치에 따른 응답

    try:
        msg = int(msg)  # 응답이 숫자이면 함수와 매칭할 예정
    except:
        return msg  # 응답이 문자열이면 바로 출력한다.

    # 만약에 응답이 숫자이면 함수 매칭
    if msg in response_functions:
        msg = response_functions[msg](user_input)  # 함수호츌

    return msg  # 5.


def main(user_input):
    print(user_input)
    result = response(user_input)  # 입력받은 내용을 함수에 넣어 응답을 예측를 한다.
    return result

# if __name__ == "__main__":
#     text = "여기는 뭐하는 곳이야"
#     result = main(text)
#     print(result)
