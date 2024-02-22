### tokenization
import pandas as pd
from collections import defaultdict
from konlpy.corpus import stopwords
from konlpy.tag import Mecab
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


path = "C:/Users/yis82/OneDrive/Desktop/Lunch Lab/data/"
user_df = pd.read_excel(path + "user_df.xlsx")

# defaultdict 사용하여 각 유저 ID에 대한 정보를 리스트로 저장
user_info_dict = defaultdict(list)

for index, row in user_df.iterrows():
    user_id = row["아이디"]
    # 직접 리뷰를 토큰화
    user_info = {
        "매장명": row["매장명"],
        "카테고리": row["카테고리"],
        "주소": row["주소"],
        "리뷰": row["리뷰 내용"],  # 토큰화된 리뷰 사용
        "방문일자": row["방문일자"],
        "요일": row["요일"],
    }
    # 해당 user_id에 대한 정보 리스트에 추가
    user_info_dict[user_id].append(user_info)

user_info_dict = dict(user_info_dict)
# 한국어에서 불용어를 제거하는 방법으로는 간단하게는 토큰화 후에 조사, 접속사 등을 제거하는 방법이 있습니다.
# 하지만 불용어를 제거하려고 하다보면 조사나 접속사와 같은 단어들뿐만 아니라
# 명사, 형용사와 같은 단어들 중에서 불용어로서 제거하고 싶은 단어들이 생기기도 합니다.
# 결국에는 사용자가 직접 불용어 사전을 만들게 되는 경우가 많습니다.
# stopwords  = 정해야 함.
mecab = Mecab()

# kykim/bert-kor-base은 한국어 BERT 모델. 한국어 자연어 처리 연구를 위해 한국어로 학습된 언어모델.
# ‘BertForSequenceClassification’은 BERT 모델을 텍스트 시퀀스 분류 작업에 맞게 미세 조정하기 위한 모델 클래스.
model_name = "kykim/bert-kor-base"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)


def classify_emotion(text):
    # 텍스트 토큰화 및 패딩
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # 예측 수행


def encode(sents, tokenizer):
    input_ids = []
    attention_mask = []

    for text in sents:
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=20,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        input_ids.append(tokenized_text["input_ids"])
        attention_mask.append(tokenized_text["attention_mask"])
    return tf.convert_to_tensor(input_ids, dtype=tf.int32), tf.convert_to_tensor(
        attention_mask, dtype=tf.int32
    )
