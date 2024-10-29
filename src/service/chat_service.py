from flask import request, abort, jsonify
import json
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from src.service.salary_service import predictSalary
# 데이터 전처리
from konlpy.tag import Okt
import re  # 정규표현식
# 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# [1] 데이터 준비: csv, db, 함수(코드/메모리) 등
# 1. 챗봇 질문 응답 데이터
data = pd.read_csv("service/챗봇데이터.csv")

# 2. 불용어
# https://gist.githubusercontent.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a/raw/4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt 사용
stopwords = pd.read_csv("service/stopwords-ko.txt", encoding="utf-8", header=None)[0].tolist()
# print(stopwords)


# 3. 선수 이름 리스트
# CSV 파일에서 선수 이름을 로드하는 함수
def load_player_names(filename='crawl_csv/stat2024.csv'):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        player_names = [row[11] for row in reader]  # 첫 번째 열에 선수 이름이 있다고 가정
    return player_names


player_names = load_player_names()

# [2] 데이터 전처리
inputs = list(data['Q'])  # 질문
outputs = list(data['A'])  # 응답


okt = Okt()


# 1. 질문 전처리
def preprocess(question):
    # 정규표현식 수정: 영어 알파벳 포함
    # result = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]', '', text)
    result = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', question)
    # 형태소 분석
    result = okt.pos(result)
    # 명사(Noun), 동사(Verb), 형용사(Adjective) 선택 가능
    result = [word for word, pos in result if pos in ['Noun', 'Verb', 'Adjective']]
    # 불용어 처리 (예시)
    # stop_words = ['이', '그', '저', '것', '하다']  # 필요에 따라 수정
    result = [word for word in result if word not in stopwords]
    # 최종 반환
    return " ".join(result).strip()


# 전처리 실행  # 모든 질문을 전처리 해서 새로운 리스트
processed_inputs = [preprocess(question) for question in inputs]
# print( processed_inputs )

# 3. 토크나이저

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


# [3] 모델 구성
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_sequence_length),
    Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    BatchNormalization(),
    Dropout(0.3),
    # 추가 LSTM 레이어
    Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    BatchNormalization(),
    Dropout(0.3),
    # 또 다른 LSTM 레이어 추가
    Bidirectional(LSTM(128)),  # return_sequences=False
    BatchNormalization(),
    Dropout(0.3),
    # 출력층
    Dense(len(outputs), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# [4] 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])  # 학습률 감소


# Learning Rate Scheduler 함수
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * tf.math.exp(-0.1)  # 학습률 감소
    return lr


# 3. 데이터셋 분리
input_train, input_val, output_train, output_val = train_test_split(input_sequences, output_sequences, test_size=0.2)

# 체크포인트 및 조기 중단 설정
checkpoint_path = 'best_performed_model.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor='loss', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=5)

# 학습
# TODO: test-train split을 제거하고 loss 함수 기준으로만 early_stop, ckpt 파일 대신 .weight.h5 및 체크포인트 파일 사용하는 분기?
# TODO: 정확도가 낮을 때 gemini API로 보내기?
# TODO: python 3.8 수업 버전으로 써보기
batch_size = 16  # 원하는 배치 크기로 설정
history = model.fit(input_train, output_train, validation_data=(input_val, output_val),
                    callbacks=[checkpoint, early_stop],
                    epochs=20,
                    batch_size=batch_size)  # 배치 크기 지정

model.summary()


# 4. 예측하기
def response(user_input):
    text = preprocess(user_input)  # 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences([text])  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences(text, maxlen=max_sequence_length)
    predict = model.predict(text)  # 3. 예측
    print("predict: ", predict)
    max_index = np.argmax(predict)  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    confidence = predict[0][max_index]  # 예측 확률

    # 예측 확률이 특정 임계값 이하일 경우
    if confidence < 0.5:  # 예: 0.5 이하일 때
        print("예측의 정확도가 낮습니다. 다른 질문을 해보세요.")  # 콘솔 출력
        return None  # 함수 출력하지 않음

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


# 챗봇에서 실행할 함수


# {1} 연봉 검색 결과를 문장으로 반환
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


# {2} 월간 경기 일정
def month_schedule():
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


# {4} 홈페이지로 이동 (하는 주소 문자열 반환)
def redirect_home(user_input):
    return 'http://localhost:8080/'


# 응답으로 실행할 함수 dict {응답 숫자 : 실행할 함수}
response_functions = {
    1: salary,
    2: month_schedule,
    4: redirect_home
    # 3 : 게시판 글쓰기
    # 월간 경기 일정 띄우기
    #
}


# if __name__ == "__main__":
#     text = "여기는 뭐하는 곳이야"
#     result = main(text)
#     print(result)
