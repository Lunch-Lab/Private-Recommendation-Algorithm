import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from konlpy.tag import Okt

path = "C:/Users/yis82/OneDrive/Desktop/Lunch Lab/data/data"
df = pd.read_excel(path + "/user_df_test.xlsx")
users = df["아이디"]
texts = df["리뷰 내용"].tolist()

user_reviews = df.groupby("아이디")["리뷰 내용"].agg(" ".join).reset_index()

okt = Okt()


# 텍스트 데이터 전처리 함수
def preprocess_text(text):
    tokens = okt.morphs(text, stem=True)  # 형태소 분석
    return " ".join(tokens)


df["리뷰 내용"] = df["리뷰 내용"].apply(preprocess_text)
# TF-IDF 벡터라이저 초기화
tfidf_vectorizer = TfidfVectorizer(max_features=100)

# 리뷰 내용을 이용해 TF-IDF 벡터 변환
tfidf_matrix = tfidf_vectorizer.fit_transform(df["리뷰 내용"])

# KMeans 클러스터링
num_clusters = 3  # 클러스터의 수는 상황에 따라 조절
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(tfidf_matrix)

# 클러스터 결과
df["클러스터"] = km.labels_

# 사용자별로 가장 많이 속한 클러스터 확인
user_interests = df.groupby("아이디")["클러스터"].agg(lambda x: x.mode()[0])

# 각 클러스터의 중심점에 대한 단어 중요도 확인
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print("Cluster %d:" % i, end="")
    for ind in order_centroids[i, :10]:  # 각 클러스터별 상위 10개 단어 출력
        print(" %s" % terms[ind], end="")
    print()

# 이를 통해 각 클러스터가 어떤 POI나 관심사를 대표하는지 해석할 수 있습니다.
