# 2_미니챗봇.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split

# RNN 기본구조 : 1. 데이터수집 2.전처리 3.토큰화/패딩 4. 모델구축 5.모델학습 6.모델평가(튜닝) 7.모델예측
# 현재 시간을 구하는 서비스 함수 호출
def time_info():
    print('현재 시간 서비스 실행')
    result = '3시' # 현재 시간을 구하는 로직하는 함수 호출
    return f'현재 시간은 {result} 입니다.' # 챗봇이 응답하는 메시지 구성 반환

def stock_info(*kwargs): # 여러개의 매개변수를 받기 # 가변길이의 매개변수 : *매개변수명(튜플) , **매개변수명(딕셔너리)
    # 클라의 재고를 알려줘 => 우리의 제품목록중에 동일한 제품명이 있는지 확인
    print('현재 재고 서비스 실행')
    result = 30 # 현재 OO의 제품 재고를 구하는 로직하는 함수 호출 # 자바 REST 호출 함수
    if result == False:
        return '제품을 확인하기 위해서 제품명을 정확히 알려주세요.'
    return f'{"콜라"}의 재고는 {result}입니다'

# 예측한 확률의 질문과 함수 매칭 딕셔너리
response_functions = {
    0 : time_info ,  # () 제외 # 지금 몇시에요? 라는 예측 질문을 찾았을 떄 함수 실행
    5 : stock_info
}

# 1. 데이터 수집 # csv , db , 함수(코드/메모리)
data = pd.read_csv("챗봇데이터.csv")
# print( data )

# 2. 데이터 전처리
inputs = list( data['Q'] ) # 질문
outputs = list( data['A'] ) # 응답

from konlpy.tag import Okt
import re # 정규표현식
okt = Okt()


def preprocess(text):
    # 정규표현식 수정: 영어 알파벳 포함
    result = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]', '', text)
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
processed_inputs = [ preprocess(질문) for 질문 in inputs ]
# print( processed_inputs )

# 3. 토크나이저
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(filters='' , lower=False , oov_token='<OOV>') # 변수명=클래스명()
tokenizer.fit_on_texts( processed_inputs ) # 전처리된 단어 목록을 단어사전 생성
# print( tokenizer.word_index ) # 사전확인

input_sequences = tokenizer.texts_to_sequences( processed_inputs ) # 벡터화
# print( input_sequences )

max_sequence_length = max( len(문장) for 문장 in input_sequences ) # 여러 문장중에 가장 긴 단어의 개수
# print( max_sequence_length ) # '좋은 책 추천 해 주세요' # 5

input_sequences = pad_sequences( input_sequences  , maxlen=max_sequence_length ) # 패딩화 # 가장 길이가 긴 문장 기준으로 0으로 채우기
# print( input_sequences ) #  '오늘 날씨 어때요' --> [ 2  3  4 ] --> [ 0 0 2 3 4 ] # 좋은 성능을 만들기 위해 차원을 통일

# 종속변수 # 데이터프레임 --> 일반 배열 변환
# output_sequences = np.array(  outputs  )
# print( output_sequences )
output_sequences = np.array( range( len( outputs ) ) )
# print( output_sequences )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM , Dense , Bidirectional , Dropout

# 모델
model = Sequential()
print(tokenizer.word_index)
model.add(Embedding(input_dim=len(tokenizer.word_index), output_dim=50, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(512, return_sequences=True , kernel_regularizer=tf.keras.regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# 추가 LSTM 레이어
model.add(Bidirectional(LSTM(256, return_sequences=True , kernel_regularizer=tf.keras.regularizers.l2(0.01))))  # return_sequences=True를 통해 다음 LSTM 레이어에 시퀀스 전달
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
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])  # 학습률 감소

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
                    epochs=200,
                    batch_size=batch_size)  # 배치 크기 지정

# 4. 예측하기
def response( text ) :
    text = preprocess( text )# 1. 예측할 값도 전처리 한다.
    text = tokenizer.texts_to_sequences( [ text ] )  # 2. 예측할 값도 토큰 과 패딩  # 학습된 모델과 데이터 동일
    text = pad_sequences( text , maxlen= max_sequence_length )
    result = model.predict( text ) # 3. 예측
    max_index = np.argmax( result )  # 4. 결과 # 가장 높은 확률의 인덱스 찾기
    msg = outputs[max_index]

    # 만약에 예측한 질문의 인덱스가 함수 매칭 딕셔너리내 존재하면
    if max_index in response_functions:
        msg += response_functions[max_index]()  # 함수호츌

    return msg  # 5.

def main(text):
    print(text)
    result = response(text)  # 입력받은 내용을 함수에 넣어 응답을 예측를 한다.
    return result

# if __name__ == "__main__":
#     text = "여기는 뭐하는 곳이야"
#     result = main(text)
#     print(result)