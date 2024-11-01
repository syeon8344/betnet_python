import os

from flask import request, abort, jsonify
import json
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 성장 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
import matplotlib.pyplot as plt

# [1] 데이터 준비: csv, db, 함수(코드/메모리) 등
# 1. 챗봇 질문 응답 데이터
data = pd.read_csv("service/챗봇데이터.csv", header=0)

# 2. 불용어
# https://gist.githubusercontent.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a/raw/4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt 사용
stopwords = pd.read_csv("service/stopwords-ko.txt", encoding="utf-8", header=None)[0].tolist()


# ======= 시각화 테스트 ===========
# 인덱스 구간마다 같은 값을 가지는 CSV이므로 응답과 예측값을 실제 값으로 나타낼 수 있지 않을까?
# 특정 열의 이름 (예: 'column_name')을 설정
column_name = 'your_column_name'

# 같은 값의 연속 구간 찾기
print(type(data['A']))
print(data['A'].dtype)
data["group"] = (data["A"] != data["A"].shift()).cumsum()
print(data["group"])

result = data.groupby('group')['A'].first().reset_index()
print("\n각 그룹의 첫 번째 값:")
print(result)


# 그룹화하여 각 구간의 첫 번째 값과 인덱스 리스트 가져오기
result = data.groupby('group').agg(
    first_value=('A', 'first'),  # 각 구간의 첫 번째 값
    index_list=('A', lambda x: list(data.index[x.index]))  # 각 구간에 해당하는 인덱스 리스트
).reset_index(drop=True)

print(result)
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

# # data 데이터프레임의 데이터 섞기
# data = data.sample(frac=1, random_state=7).reset_index(drop=True)  # frac: 섞을 비율, 1이므로 전체 데이터 섞기
# 2. 데이터 전처리
inputs = list(data['Q'])  # 질문
outputs = list(data['A'])  # 응답


okt = Okt()


# 1. 질문 전처리
def preprocess(text):
    # 정규표현식 수정: 영어 알파벳 포함
    # result = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]', '', text)
    result = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', text)
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
processed_inputs = [preprocess(question) for question in list(data['Q'])]

# 3. 토크나이저

tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')  # 변수명=클래스명()
tokenizer.fit_on_texts(processed_inputs)  # 전처리된 단어 목록을 단어사전 생성
print(tokenizer.word_index)  # 사전확인

input_sequences = tokenizer.texts_to_sequences(processed_inputs)  # 벡터화

max_sequence_length = max(len(sentence) for sentence in input_sequences)  # 여러 문장중에 가장 긴 단어의 개수
# print( max_sequence_length ) # '좋은 책 추천 해 주세요' # 5

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length)  # 패딩화 # 가장 길이가 긴 문장 기준으로 0으로 채우기
# print( input_sequences ) #  '오늘 날씨 어때요' --> [ 2  3  4 ] --> [ 0 0 2 3 4 ] # 좋은 성능을 만들기 위해 차원을 통일
# training_size = int(len(data) * 0.8)  # 전체 데이터에서 훈련용 데이터의 비율
# train_input_seq = input_sequences[:training_size]  # 전체 데이터에서 훈련 데이터 비율까지 슬라이싱
# valid_input_seq = input_sequences[training_size:]  # 나머지는 테스트 데이터 슬라이싱

# 종속변수 # 데이터프레임 --> 일반 배열 변환
# output_sequences = np.array(  outputs  )
# print( output_sequences )
output_sequences = np.array(range(len(outputs)))
# train_output = output_sequences[:training_size]
# valid_output = output_sequences[training_size:]

# 1. 전체 데이터의 인덱스 생성  # [0, 1, 2, 3, ... ]
total_indices = np.arange(len(data))

# 2. 랜덤으로 20%의 인덱스 선택
test_size = int(0.2 * len(data))
test_indices = np.random.choice(total_indices, size=test_size, replace=False)

# 3. 훈련 데이터셋 및 검증 데이터셋 생성  # np.setdiff1d: A 배열에서 B 배열의 값들을 뺀 배열을 반환
train_indices = np.setdiff1d(total_indices, test_indices)  # 배열 내 값 == 인덱스들

# 4. 훈련용 질문 및 응답
processed_train_inputs = np.array([input_sequences[i]for i in train_indices])
print(type(processed_train_inputs))
print(type(processed_train_inputs[0]))
print(processed_train_inputs[:3])
processed_train_outputs = np.array([output_sequences[i] for i in train_indices])
print(type(processed_train_outputs))
print(type(processed_train_outputs[0]))

# 5. 테스트용 질문 및 응답
processed_test_inputs = np.array([input_sequences[i] for i in test_indices])
print(type(processed_test_inputs))
print(type(processed_test_inputs[0]))
print(processed_test_inputs[:3])
processed_test_outputs = np.array([output_sequences[i] for i in test_indices])
print(type(processed_test_outputs))
print(type(processed_test_outputs[0]))

# [3] 모델 구성
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=200, input_length=max_sequence_length),
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
              metrics=['accuracy'])


# Learning Rate Scheduler 함수
def scheduler(epoch, lr):
    if epoch > 5:
        return lr * tf.math.exp(-0.1)  # 학습률 감소
    return lr


# # 3. 데이터셋 분리
# 체크포인트 콜백 설정 (가중치만 저장)
checkpoint = ModelCheckpoint('ballgpt_model_weights.h5', save_weights_only=True, save_best_only=True)
# val_loss 수치 갱신이 일정 회수 이상 되지 않으면 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
# 일정 에포크 이상부터 학습률 감소 (에포크가 진행될수록 세부 조정의 효과)
lr_scheduler = LearningRateScheduler(scheduler)


# if os.path.exists('ballgpt_model_weights.h5'):
#     model.load_weights('ballgpt_model_weights.h5')
# else:
# 학습
batch_size = 64  # 원하는 배치 크기로 설정
# 전체 챗봇데이터로 훈련하고 검증은 20퍼센트 샘플 추출해서 진행
"""
input_sequences: 패딩까지 완료된 전체 질문 시퀀스
output_sequences: 인덱스 번호 = 값인 배열 [0, 1, 2, ... ] 
-> 예측된 값은 1163칸을 갖는 배열, 그 중 가장 큰 값이 예측된 인덱스 값 -> CSV의 인덱스를 참고하면 해당 예측 답 해석 가능.
validation_data=(
    processed_test_inputs,  # 질문-응답 연결을 깨지 않기 위해 인덱스 수준에서 랜덤 샘플링해서 검증 질문-응답 셋을 만들었다.
    processed_test_outputs
    )
"""
history = model.fit(input_sequences, output_sequences, validation_data=(processed_test_inputs, processed_test_outputs),
                    callbacks=[checkpoint, early_stop, lr_scheduler],
                    epochs=200,
                    batch_size=batch_size)  # 배치 크기 지정

model.summary()
# TODO: 응답이 같은 인덱스들이 있다는 것을 사용해서 그룹화하면 좋지 않을까?

# # ============ 시각화 ===============
# import matplotlib.font_manager as fm
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
#
# # 폰트 설정
# font_path = "C:/Windows/Fonts/malgun.ttf"
# font_prop = fm.FontProperties(fname=font_path)
#
# # 마이너스 기호 문제 해결하기
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family=font_prop.get_name())
#
#
# def visualize_model_performance(train_output, output_classes):
#     # Confusion Matrix 혼동 매트릭스
#     """
#                         예측된 긍정 (Positive)	예측된 부정 (Negative)
#     실제 긍정 (Positive)	    진양성 (TP)	            거짓음성 (FN)
#     실제 부정 (Negative)	    거짓양성 (FP)	        진음성 (TN)
#     """
#     print("CM")
#     cm = confusion_matrix(train_output, output_classes)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
#     plt.xlabel('예측된 클래스')
#     plt.ylabel('실제 클래스')
#     plt.title('혼동 행렬')
#     plt.show()
#
#     # Classification Report
#     report = classification_report(train_output, output_classes)
#     print("분류 보고서:\n", report)
#
#     # 실제 값과 예측 값의 산점도
#     print("Scatter")
#     plt.figure(figsize=(10, 6))
#     plt.scatter(train_output, output_classes, alpha=0.6)
#     plt.plot([train_output.min(), train_output.max()], [train_output.min(), train_output.max()], 'k--', lw=2)  # y=x 선
#     plt.xlabel('실제 값')
#     plt.ylabel('예측 값')
#     plt.title('실제 값 vs 예측 값')
#     plt.grid(True)
#     plt.show()
#
#     # 예측 확률 분포 히스토그램
#     print("histogram")
#     plt.figure(figsize=(10, 6))
#     plt.hist(output_classes, bins=30, alpha=0.6, label='예측된 클래스', color='orange')
#     plt.hist(train_output, bins=30, alpha=0.6, label='실제 클래스', color='blue')
#     plt.xlabel('클래스 레이블')
#     plt.ylabel('빈도')
#     plt.title('실제 클래스 vs 예측 클래스 분포')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # 예시로 모델 예측 수행 후 시각화
# output_predict = model.predict(train_input_seq)
# output_classes = np.argmax(output_predict, axis=1)  # 예측 클래스
# print("visualizing")
# print("output_classes: ", output_classes.shape)
# print("train_output: ", train_output.shape)
# print("Unique classes in output_classes:", np.unique(output_classes))
# print("Unique classes in train_output:", np.unique(train_output))
# visualize_model_performance(train_output, output_classes)
#
#
# # 손실 함수 시각화
# def plot_loss(history):
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# plot_loss(history)

# =========== 시각화 ================


# 4. 예측하기
def response(user_input):
    text = preprocess(user_input)  # 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences([text])  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences(text, maxlen=max_sequence_length)
    predict = model.predict(text)  # 3. 예측
    print("predict: ", predict)
    print(predict.shape())
    max_index = np.argmax(predict)  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    # confidence = predict[0][max_index]  # 예측 확률
    #
    # # 예측 확률이 특정 임계값 이하일 경우
    # if confidence < 0.5:  # 예: 0.5 이하일 때
    #     print("예측의 정확도가 낮습니다. 다른 질문을 해보세요.")  # 콘솔 출력
    #     return None  # 함수 출력하지 않음
    print("max_index: ", max_index)
    msg = outputs[max_index]  # max_index : 예측한 질문의 위치 . # msg : 예윽한 질문의 위치에 따른 응답

    try:
        msg = int(msg)  # 응답이 숫자이면 함수와 매칭할 예정
    except:
        return msg  # 응답이 문자열이면 바로 출력한다.

    # 만약에 응답이 숫자이면 함수 매칭
    if msg in response_functions:
        try:
            msg = response_functions[msg](user_input)  # 함수호츌
        except TypeError:
            msg = response_functions[msg]

    return msg  # 5.


# 챗봇에서 실행할 함수 (controller에서 jsonify가 적용되므로 파이썬 리스트, 딕셔너리 등을 반환)
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


# {2} 이번달 월간 경기 일정
def month_schedule(user_input):
    # # 쿼리 문자열에서 year와 month 가져오기
    # year = request.args.get('year', type=int)
    # month = request.args.get('month', type=int)
    #
    # if year and month:
    #     date = f'{year:04d}{month:02d}'
    # else:
    #     date = pd.to_datetime('today').strftime('%Y%m')  # 날짜 입력이 없을 시 현재 년도 월
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
    # print(json.loads(df.to_json(orient='records', force_ascii=False)))
    return df.to_dict(orient='records')


# {3} 뉴스 목록 띄우기
def get_news(user_input):
    pass


# {4} 홈페이지로 이동 (하는 주소 문자열 반환)
def redirect_home(user_input):
    return 'http://localhost:8080/'


# {5} 뉴스 이동
def redirect_news(user_input):
    return 'http://localhost:8080/article'


# {6} 굿즈마켓 이동
def redirect_market(user_input):
    return 'http://localhost:8080/market'


# {7} 설문조사 이동
def redirect_poll(user_input):
    return 'http://localhost:8080/poll'


# {8} 게시판 이동
def redirect_board(user_input):
    return 'http://localhost:8080/board'


# {9} 제미니 이동
def redirect_gemini(user_input):
    return 'http://localhost:8080/gemini'


# {10} 타자기록 이동
def redirect_hitter(user_input):
    return 'http://localhost:8080/hitter'


# {11} 투수기록 이동
def redirect_pitcher(user_input):
    return 'http://localhost:8080/pitcher'


# {12} 주루기록 이동
def redirect_runner(user_input):
    return 'http://localhost:8080/runner'


# {13} 순위기록 이동
def redirect_rank(user_input):
    return 'http://localhost:8080/rank'


# 응답으로 실행할 함수 dict {응답 숫자 : 실행할 함수}
response_functions = {
    1: salary,
    2: month_schedule,
    3: get_news,
    4: redirect_home,
    5: redirect_news,
    6: redirect_market,
    7: redirect_poll,
    8: redirect_board,
    9: redirect_gemini,
    10: redirect_hitter,
    11: redirect_pitcher,
    12: redirect_runner,
    13: redirect_rank,

    # 3 : 게시판 글쓰기
    # 월간 경기 일정 띄우기
    #
}


# 웹 페이지에서 입력을 받아 모델에서 출력한 값을 반환
def main(user_input):
    # print(user_input)
    result = response(user_input)  # 입력받은 내용을 함수에 넣어 응답을 예측를 한다.
    return result


# if __name__ == "__main__":
#     text = "여기는 뭐하는 곳이야"
#     result = main(text)
#     print(result)
#     손실 값 시각화

