### tokenization
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize

path = "C:/Users/yis82/OneDrive/Desktop/Lunch Lab/data/"
user_df = pd.read_excel(path + "user_df.xlsx")

# defaultdict 사용하여 각 유저 ID에 대한 정보를 리스트로 저장
user_info_dict = defaultdict(list)

for index, row in user_df.iterrows():
    user_id = row["아이디"]
    # 직접 리뷰를 토큰화
    tokenized_review = word_tokenize(row["리뷰 내용"])
    user_info = {
        "매장명": row["매장명"],
        "카테고리": row["카테고리"],
        "주소": row["주소"],
        "리뷰": tokenized_review,  # 토큰화된 리뷰 사용
        "방문일자": 
    }
    # 해당 user_id에 대한 정보 리스트에 추가
    user_info_dict[user_id].append(user_info)

user_info_dict = dict(user_info_dict)
