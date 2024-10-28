# day39 > 1_시퀀스2시퀀스.py
import pandas as pd
import re
# 챗봇 질문 응답 데이터
from Korpora import KoreanChatbotKorpus
# 형태소 분석기
from konlpy.tag import Okt
# 단어사전 토큰화
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 모델 학습
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# 챗봇 라이브러리 불러오기
# 소스에서 URL 오류, 'url': 'https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv'로 변경
corpus = KoreanChatbotKorpus()
print(type(corpus.train))  # <class 'Korpora.korpora.LabeledSentencePairKorpusData'> DataFrame과 비슷하다
corpus_df = pd.DataFrame(corpus.train)
print(corpus_df.head())

# 챗봇 데이터 샘플 확인
print(corpus.get_all_texts()[:5])  # 질문 "열"의 상단 5개 데이터 확인
print(corpus.get_all_pairs()[:5])  # 응답 "열"의 상단 5개 데이터 확인

# text와 pair 쌍
print("Q: ", corpus.train[0].text)  # Q:  12시 땡!
print("A: ", corpus.train[0].pair)  # A:  하루가 또 가네요.

# 전체 코퍼스 크기 11823
print("Total Corpus Size: ", len(corpus.get_all_texts()))
print(corpus_df.shape)  # .shape: DataFrame 객체의 차원 확인, (11823, 3)

# 1000개 샘플링
texts = []
pairs = []
# for index, value in enumerate(list/tuple)
# for value in list/tuple
for i in range(1500):
    # print("Q: ", corpus_df["text"][i])
    # print("A: ", corpus_df["pair"][i])
    texts.append(corpus.train[i].text)
    pairs.append(corpus.train[i].pair)

print(texts[:5])
print(pairs[:5])


# 2. 데이터 전처리
# 정규식 regex로 데이터 전처리 함수
def clean_sentence(sentence):
    # 한글, 숫자, 띄어쓰기를 제외한 모든 문자를 제거, re.sub(): 파이썬 내장 문자열 정규표현식 함수
    # pd['열이름'].str.replace(정규식, 변환 결과 문자, regex=True): DataFrame 내 정규표현식 적용
    sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', sentence)  # ' ' == \s
    return sentence


# 전처리 함수 테스트
print(clean_sentence("텐서플로!@#$%^"))  # 텐서플로
print(clean_sentence("하루가 또 가네요."))  # 하루가 또 가네요

# konlpy > Okt 형태소
okt = Okt()  # 한글 분석기 개체


def process_morph(sentence):
    return ' '.join(okt.morphs(sentence))  # 사이에_추가할_문자열.join(문자열 리스트)


# Seq2Seq 모델 훈련시 필요 데이터셋: 전체 질문, 답변 시작, 답변 끝
# 인코더 question(질문 전체), 디코더에 입력 answer_input(<START>를 앞에 추가), 디코더 출력할 answer_output(<END>를 끝에 추가)
# 한글 문장 전처리
def clean_and_morph(sentence, is_question=True):  # 매개변수명=초기값: 매개변수의 초기값 설정
    # 한글 문장 전처리
    sentence = clean_sentence(sentence)
    # konlpy > Okt 형태소 변환
    sentence = process_morph(sentence)
    # 질문인 경우와 답변인 경우 분기
    if is_question:
        return sentence
    else:
        # START 토큰은 decoder input, END 토큰은 decoder output에 추가
        return "<START> " + sentence, sentence + " <END>"  # 소괄호가 생략된 튜플 반환 (함수의 연산 결과는 항상 1개)


def preprocess(texts, pairs):
    questions = []
    answer_in = []
    answer_out = []
    # 질의 전처리
    for text in texts:
        # 전처리와 morph 수행
        question = clean_and_morph(text, is_question=True)
        questions.append(question)
    # 답변 전처리
    for pair in pairs:
        # pair: (입력 답변, 출력 답변)
        in_, out_ = clean_and_morph(pair, is_question=False)
        answer_in.append(in_)
        answer_out.append(out_)
    return questions, answer_in, answer_out


questions, answer_in, answer_out = preprocess(texts, pairs)
print(questions[:2])  # ['12시 땡', '1 지망 학교 떨어졌어']
print(answer_in[:2])  # ['<START> 하루 가 또 가네요', '<START> 위로 해 드립니다']
print(answer_out[:2])  # ['하루 가 또 가네요 <END>', '위로 해 드립니다 <END>']

# 전체 문장을 하나의 리스트로 만들기
all_sentences = questions + answer_in + answer_out

# 단어 사전 만들기
# filters='': 토큰화 과정에서 특정 기호를 제거(필터)
# lower=False: 영문을 소문자로 전부 변환할지 (기본값 True: 전부 소문자로)
# oov_token: 단어 사전에 없는 문자를 매칭하면 <OOV> 문자열로 표현
tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(all_sentences)
print(tokenizer.word_index)  # 단어 사전 확인
# {'<OOV>': 1, '<START>': 2, '<END>': 3, '이': 4, '거': 5, '을': 6, '가': 7, '나': 8, '예요': 9, ... }
print(len(tokenizer.word_index))  # 단어의 총 개수 2300

# .texts_to_sequences(): 등록된 단어사전에 따라 문장의 단어들을 벡터(숫자) 매칭하여 변환
# 치환: 텍스트를 시퀀스로 인코딩
question_sequence = tokenizer.texts_to_sequences(questions)
print(question_sequence[0])  # [1608, 1609]
answer_in_sequence = tokenizer.texts_to_sequences(answer_in)
print(answer_in_sequence[0])  # [2, 391, 7, 356, 1234]
answer_out_sequence = tokenizer.texts_to_sequences(answer_out)
print(answer_out_sequence[0])  # [391, 7, 356, 1234, 3]
# 문장의 길이 맞추기 (pad_sequences)
MAX_LENGTH = 30  # 임의값
question_padded = pad_sequences(question_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
answer_in_padded = pad_sequences(answer_in_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
answer_out_padded = pad_sequences(answer_out_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')

print(question_padded.shape, answer_in_padded.shape, answer_out_padded.shape)  # (1000, 30) (1000, 30) (1000, 30)


# 인코더
# 상속: 하나의 클래스가 다른 클래스에게 속성/필드와 함수/기능을 물려주는 행위
# JAVA: class A extends B, this, super
# python: class A(B), self, super
class Encoder(tf.keras.Model):
    # 초기화 함수 (생성자): 사용할 변수, 레이어를 미리 불러와 파라미터 값들을 설정
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        # 1. units: LSTM에서 사용할 유닛/노드/뉴런 수
        # 2. vocab_size: 임베딩 레이어로 들어가는 단어 크기, 문장 내 단어마다 +1
        # 3. embedding_dim: 임베딩 레이어의 각 단어 출력 벡터의 차원 (한 단어를 몇 차원으로 구성할 것인가?)
        # 4. time_steps: 임베딩 레이어 입력으로 들어가는 시퀀스 길이 (한번에 학습할 단어 수)
        # super(): 상속하는 부모 클래스, .__init__(): 생성자를 불러온다 -> Model 클래스의 모든 속성 이용 가능.
        super(Encoder, self).__init__()
        # 1. Embed 레이어
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps)
        # 2. 드롭아웃 레이어: 매 회차마다 매개변수 비율만큼 랜덤하게 학습 X
        self.dropout = Dropout(0.2)
        # 3. LSTM 레이어
        self.lstm = LSTM(units, return_state=True)

    # 실행 함수 .call()
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x)
        # x: 현재 문장의 특징/정보/패턴
        # 은닉 상태: LSTM 층에서 현재 + 바로 직전 시점까지 기록한 특징/정보/패턴 (단기기억 STM)
        # 셀 상태: LSTM 알고리즘이 전체 단어들에서 습득한 중요 특징/정보/패턴 (장기기억 LTM)
        # 특징, 정보, 패턴?
        # CNN: 이미지 분석 - 곡선, 색감, 사이즈, 비율, 질감(텍스처) 등 (RGB 0~255)
        # RNN: 텍스트 분석 - 빈도, 형태소(동사, 형용사), 감정, 단어의 의미 등 (텍스트 대신 벡터 밀집 행렬)
        # 인코더는 입력 과정이므로 출력층 Dense가 없다
        return [hidden_state, cell_state]


# 디코더 (출력 단계)
class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        # return_state: 기본값 True, 은닉 상태와 셀 상태를 반환
        # return_sequences: 모든 시점의 출력을 반환
        self.dense = Dense(vocab_size, activation='softmax')  # 최종 출력 레이어

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)  # LSTM 레이어 변수
        # initial_state: 초기화 상태 속성, 인코더와 결합 이후 인코더가 생성한 은닉 상태와 셀 상태를 대입
        x = self.dense(x)  # 출력층의 값 "x"
        return x, hidden_state, cell_state  # (최종 확률값, 은닉 상태, 셀 상태)


# ================================= day 41 =====================================


# Seq2Seq Model 모델 결합
class Seq2Seq(tf.keras.Model):  # 클래스 정의, (): 상속
    # 1. 초기화 함수, 객체 생성시
    def __init__(self, units, vocab_size, embedding_dim, time_steps, start_token, end_token):
        super(Seq2Seq, self).__init__()
        self.start_token = start_token  # 객체 속성 정의 후 매개변수 대입. 시작 토큰: 모델 문장 생성시 시작 부분을 식별
        self.end_token = end_token  # 끝 토큰: 문장 생성시에 끝마침 식별
        self.time_steps = time_steps
        self.encoder = Encoder(units, vocab_size, embedding_dim, time_steps)  # Encoder/Decoder: 각 객체 생성 및 대입
        self.decoder = Decoder(units, vocab_size, embedding_dim, time_steps)

    # 2. 실행 함수, 객체 호출시
    def call(self, inputs, training=True):
        # inputs: 모델 객체로 들어오는 입력 데이터
        # training=True: t/f 훈련중/훈련아님. 기본값은 =True
        if training:  # .fit() 훈련중일 경우
            encoder_inputs, decoder_inputs = inputs  # 현재 모델의 입력 -> 인코더, 디코더 입력, fit()시 들어오는 inputs 데이터
            context_vector = self.encoder(encoder_inputs)  # 인코더 객체의 호출 함수 (call) 적용 후 결과
            # python) _(언더바) -> 변수 생략. e.g. for _ in list ..., initial_state: 인코더 결과 값
            (decoder_outputs, _, _) = self.decoder(inputs=decoder_inputs, initial_state=context_vector)
            return decoder_outputs  # 디코더 출력 반환, 최종 예측 확률값
        else:  # 예측/추론 모드, 문장 생성/예측 모드
            context_vector = self.encoder(inputs)
            # 시작 토큰을 이용한 2차원 텐서 생성, tf.constant(): 텐서 생성
            target_seq = tf.constant([[self.start_token]], dtype=tf.float32)
            # tf.TensorArray(): 텐서 배열 생성, 디코더 출력 결과를 저장하는 배열 생성하기
            results = tf.TensorArray(tf.int32, self.time_steps)

            # 디코더가 다음 단어(-> 문장 생성)를 예측하는 과정을 반복
            for i in tf.range(self.time_steps):
                # (최종 확률값, 은닉 상태, 셀 상태) = 디코더 객체 호출 함수의 결과
                decoder_output, decoder_hidden, decoder_cell = self.decoder(target_seq, initial_state=context_vector)
                # 예측 결과에서 가장 높은 확률의 인덱스 찾기 tf.argmax(): 가장 높은 값의 인덱스
                decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1), dtype=tf.int32)
                # 차원 조정: 2차원 값 1개짜리 결과값으로
                decoder_output = tf.reshape(decoder_output, shape=(1, 1))
                # (TensorArray).write(): i번째 인덱스의 예측한 단어를 텐서배열에 저장
                results = results.write(i, decoder_output)
                # 예측값이 종료 토큰이면 루프 종료
                if decoder_output == self.end_token:
                    break
                # 예측값이 종료 토큰이 아니면 현재 예측한 단어를 다음 예측 시 사용
                target_seq = decoder_output
                # 디코더의 상태를 업데이트하고 다음 반복에 사용
                context_vector = [decoder_hidden, decoder_cell]
            # 반복문 종료시 모든 예측 결과를 스택으로 반환 - 모든 시퀀스 예측한 결과가 포함
            # 시퀀스: 문장의 단어를 정해진 순서대로 나열한 것, e.g. A B C -> ["A", "B", "C"]
            return tf.reshape(results.stack(), shape=(1, self.time_steps))


# 디코더의 결과를 원 핫 인코딩 벡터로 변환하기
# tokenizer.word_index: 단어사전, + 1은 <OOV>까지 포함해야 하므로 추가
VOCAB_SIZE = len(tokenizer.word_index) + 1

"""
컴퓨터가 이해하는 언어인 벡터로 변환하는 방법
e.g. [1, 0, 0, 0] 희소 행렬 <-> [0.9, 0.1] 밀집 행렬
임베딩(밀집행렬): <-> 원 핫 벡터(희소 행렬), 주로 챗봇 질문에 사용 (학습 데이터), 단어간의 유사성 파악에 유리하다 
원핫벡터:  주로 챗봇 답변에 사용 (결과 데이터) -> 유사성 파악이 아닌 단순 분류에 유리
"""


def convert_to_one_hot(padded):
    # 원 핫 인코딩 초기화, np.zeros((shape)): 차원 구성대로 0으로 채운 배열
    # np.zeros(3): [0 0 0], np.zeros(2,3): [[0 0 0] [0 0 0]], np.zeros(2, 2, 3): [[[0 0 0] [0 0 0]] [[0 0 0] [0 0 0]]]
    one_hot_vector = np.zeros((len(answer_out_padded), MAX_LENGTH, VOCAB_SIZE))
    # (데이터 1, 데이터 2, 데이터 3): 3차원 배열 초기화
    # len(aswer_out_padded): 총 응답의 개수 (1001, 30) -> (단어사전 + <OOV>, 임의의 문장 최대치)
    # MAX_LENGTH: 문장의 최대 길이
    # VOCAB_SIZE: 단어 사전의 단어 수
    # -> (응답 단어의 총 갯수, 문장 최대 길이, 단어사전의 단어 수)

    # 디코더 목표를 원 핫 인코딩으로 변환: 지정한 인덱스마다 1을 채우기
    # 1. 행
    # 학습시 입력은 인덱스, 출력은 원 핫 인코딩
    for i, sequence in enumerate(answer_out_padded):  # for index, value in enumerate(list)
        # 2. 열
        # i: 현재 시퀀스의 인덱스, sequence: 현재 시퀀스의 단어
        for j, index in enumerate(sequence):
            # 3. 높이
            # j: 현재 단어의 인덱스 번호
            one_hot_vector[i, j, index] = 1

    return one_hot_vector


# 최대 30개의 단어 수로 패딩처리된 문장 1000개 -> 각 단어마다 2300개의 요소가 있는 단어사전의 인덱스에 따른 원-핫 인코딩 벡터
# 30: 1000개의 문장마다의 패딩 포함 단어들, 2301: 각 단어마다 단어사전을 참고하여 1 한개, 0 2300개 원 핫 인코딩
answer_in_one_hot = convert_to_one_hot(answer_in_padded)  # (1000, 30) -> (1000, 30, 2301)
answer_out_one_hot = convert_to_one_hot(answer_out_padded)  # (1000, 30) -> (1000, 30, 2301)
print(answer_in_one_hot.shape, answer_out_one_hot.shape)  # (1000, 30, 2301) (1000, 30, 2301)


# 모델(디코더 부분)이 예측한 단어 목록(indexes: 예측한 단어의 인덱스)
def convert_index_to_text(indexes, end_token):
    sentence = ""  # 생성된 문장을 저장할 변수, 빈 문자열로 시작
    # 모든 문장에 대해 반복
    for index in indexes:
        if index == end_token:
            # 끝 단어이므로 예측 준비
            break
        # 사전에 존재하는 단어이면 단어 추가
        if index > 0 and tokenizer.index_word[index] is not None:
            sentence += tokenizer.index_word[index]
        else:
            # 사전에 없는 인덱스면 빈 문자열 추가
            sentence += ""
        # 다음 단어 추가 전 띄어쓰기 공백 추가
        sentence += " "
    # 반복문 종료 -> 생성된 문장 반환
    return sentence


# 모델 객체 생성시 사용할 변수들
BUFFER_SIZE = 1000  # 1회 훈련(에포크)당 훈련 데이터에서 무작위로 선택할 최대 샘플 수, 버퍼가 클 수록 무작위성 및 학습 성능이 향상되지만 메모리 소모도 증가
BATCH_SIZE = 16  # 1회 훈련(에포크)당 1회 학습 시 사용하는 샘플 수, 클수록 안정적이지만 메모리 소모 증가. 에포크 당 총 루프 수: BUFFER_SIZE / BATCH_SIZE
EMBEDDING_DIM = 100  # 단어를 벡터로 인코딩하는 과정에서 한 단어가 임베드되는 차원 수, 클수록 표현 성능과 단어간 의미 관계 파악이 좋아지지만 메모리 소모 및 계산 비용이 증가
TIME_STEPS = MAX_LENGTH  # 임의로 설정한 문장 내 단어의 최대 길이 (30)
START_TOKEN = tokenizer.word_index['<START>']  # 문장의 시작을 알려주는 토큰 인덱스, 예측시 시작하는 위치
END_TOKEN = tokenizer.word_index['<END>']  # 문장의 끝을 알려주는 토큰 인덱스, 예측 과정에서 이 토큰을 만나면 문장 생성 종료
NUM_EPOCHS = 20   # 훈련 반복 횟수

UNITS = 128  # 유닛 수: RNN에서는 유닛, CNN에서는 노드 (== 뉴런 수), 각각 모델에서 학습 레이어에 사용되는 뉴런 수
# 유닛 수가 증가할수록 복잡한 학습이 가능하지만 과대적합 위험성이 커진다. 주로 2^n 단위 (32, 64, 128 등)

# VOCAB_SIZE = len(tokenizer.word_index) + 1, 단어 사전 수 + <OOV>
DATA_LENGTH = len(questions)  # 질문의 총 갯수
SAMPLE_SIZE = 3  # 샘플 수

# 모델 가중치를 저장하고 추후에 재사용하기
checkpoint_path = 'model/seq2seq-chatbot-checkpoint.weights.h5'  # .ckpt에서 .weights.h5로 변경
checkpoint = ModelCheckpoint(filepath=checkpoint_path,  # 모델 가중치를 저장할 경로
                             save_weights_only=True,  # 모델 구조는 빼고 가중치만 저장하기 (False시 모델 전체 저장)
                             save_best_only=True,  # 최고 결과만 저장: 성능이 향상되면 체크포인트 갱신
                             monitor='loss',  # 모니터링할 수치 (loss: 손실 함수 -> 성능 향상을 모니터링)
                             verbose=1  # 과정 로깅 수준, 생략 가능
                             )

# seq2seq 모델 생성 및 컴파일
seq2seq = Seq2Seq(UNITS, VOCAB_SIZE, EMBEDDING_DIM, TIME_STEPS, START_TOKEN, END_TOKEN)
seq2seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# 모델을 사용해서 예측하는 함수
# model: 학습한 모델, question_inputs: 예측할 질문
def make_prediction(model, question_inputs):
    # 훈련 X: 모델 내 else: 부분 실
    results = model(inputs=question_inputs, training=False)
    # 변환된 인덱스를 문장으로 변환
    results = np.asarray(results).reshape(-1)
    # .reshape(-1): 알아서 1차원 배열로 변환
    # 나중에 문장 조회시 평탄화(1차원 변경)하고 convert_index_to_text()에 전달
    return results


# 모델 훈련 및 10n 에포크 시점마다 샘플 질문 및 응답
for epoch in range(NUM_EPOCHS):  # 20회 반복
    print(f"제 {epoch*10 + 1} 에포크")
    # fit(): 모델 훈련 함수
    # 1. [question_padded, answer_in_padded]: 입력 데이터
    # 2. answer_out_one_hot: 결과 데이터
    # 3. callbacks: 훈련중 체크포인트를 지정 - 가중치만 저장
    seq2seq.fit([question_padded, answer_in_padded],
                answer_out_one_hot,
                epochs=10,  # 총합 200 에포크
                batch_size=BATCH_SIZE,
                callbacks=[checkpoint]
                )
    # 랜덤 샘플 번호, 매 훈련 루프 후 난수 생성, 난수 인덱스의 질문으로 성능 예측
    samples = np.random.randint(DATA_LENGTH, size=SAMPLE_SIZE)  # 질문 중에서 3개 샘플 질문 선택
    # 예측 성능 테스트
    for index in samples:  # 난수 인덱스로 얻은 질문 3개
        question_inputs = question_padded[index]  # 질문의 인코딩(패딩)된 단어들
        # 문장 예측, np.expand_dims(배열, 추가할 차원의 인덱스): 해당 인덱스에 차원 추가, 0이므로 맨 앞에 추가
        results = make_prediction(seq2seq, np.expand_dims(question_inputs, 0))
        # 예측한 벡터들을 문장으로 변환
        results = convert_index_to_text(results, END_TOKEN)
        # 출력
        print(f"질문: {questions[index]}")
        print(f"예측: {results}\n")
        print()


# 예측
# 자연어로 질문 입력하므로 전처리 적용
def make_question(sentence):
    sentence = clean_and_morph(sentence)
    # tokenizer.texts_to_sequences: tokenizer.word_index에서 sentence에 있는 단어들을 index로 변환
    question_sequence = tokenizer.texts_to_sequences([sentence])
    # pad_sequences: sequence에 padding(0)을 추가 또는 뒷부분을 자르기
    question_padded = pad_sequences(question_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
    return question_padded


print(make_question("오늘 날씨 어때?"))
"""
question_padded
[[237 103 360   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0]]
"""


# 챗봇
def run_chatbot(question):
    question_inputs = make_question(question)  # 질문 문장 전처리
    results = make_prediction(seq2seq, question_inputs)  # 학습된 모델과 전처리된 질문으로 응답 예측
    results = convert_index_to_text(results, END_TOKEN)  # 예측 응답 결과를 문장으로 변환한다.
    return results


# 챗봇 실행
while True:
    user_input = input("<< 말을 걸어 보세요!\n")
    if user_input == 'q':  # 'q' 입력시 종료
        break
    print(f">> 챗봇 응답: {run_chatbot(user_input)}")  # 입력받은 질문을 run_chatbot() 함수에 대입하고 예측한 문장 출력
"""
내일 날씨는?
>> 챗봇 응답: 아무 도 없는 곳 으로 여행 을 떠나 보세요 
<< 말을 걸어 보세요!
어디 가고 싶어
>> 챗봇 응답: 직접 주는 게 더 좋을 것 같아요 
<< 말을 걸어 보세요!
선물을 무엇을 주는 게 좋을까
>> 챗봇 응답: 거짓말 은 할수록 늘어요 
<< 말을 걸어 보세요!
"""