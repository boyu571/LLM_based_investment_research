import json
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

news_path = '/Users/kimkirok/Desktop/트레이디/업무/koala/dataset/koala_news/news.json'
path2 = '/Users/kimkirok/Desktop/트레이디/업무/koala/dataset/koala_news/stopword.txt'


# 코알라 뉴스 리스트로 반환
def koala_news(path):
    news = []
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f.readlines():
            a = json.loads(line)
            news.append(a)
    return news


# 기사 내용을 반환
def article(path):
    news = koala_news(path)
    article = []
    for doc in tqdm(news, total=len(news)):
        text = doc['content']
        article.append(text)
    return article


# 기사 제목을 반환
def title(path):
    news = koala_news(path)
    title = []
    for doc in tqdm(news, total=len(news)):
        text = doc['title']
        title.append(text)
    return title


# dataframe화
def doc_df(data):
    documents_df = pd.DataFrame(data, columns=['documents'])
    return documents_df


'''
    cos_sim 유사한 문서 찾기
    if 동일 사건이 많다고 예측 하는 경우 6개이상 -> 같은 사건 x라고 보고 제목을 기준으로 다시 cos_sim 측정
    결과는 좋지 않음
    
    6개 이하인 경우 같은 사건 cos_sim을 0.05씩 낮추어서 0.5까지 threshold 측정 3개까지 찾기
    다만 이 경우는 0.8대 같은사건 2개 0.6대 다른 사건 1개가 비슷하다고 예측해서 임베딩 방식을 바꿈
 
'''


def most_similar(doc_id, similarity_matrix, similarity_matrix2):
    df = doc_df(article(news_path))
    a = 0.8
    simdoc = []

    similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]

    for ix in similar_ix:
        if ix == doc_id:
            continue
        if similarity_matrix[doc_id][ix] > 0.8:
            simdoc.append(df.iloc[ix]["documents"])

    if len(simdoc) < 6:
        print(f'Document: {df.iloc[doc_id]["documents"]}')
        print('\n')
        print('Similar Documents:')
        if len(simdoc) > 1:
            for ix in similar_ix:
                if ix == doc_id:
                    continue
                if len(simdoc) > 1 and similarity_matrix[doc_id][ix] > 0.8:
                    print('\n')
                    print(f'Document: {df.iloc[ix]["documents"]}')
                    print(f'Cosine Similarity : {similarity_matrix[doc_id][ix]}')
                    print('id : ', ix + 1)

        while (len(simdoc) < 3 or a > 0.5):

            for ix in similar_ix:
                if ix == doc_id:
                    continue
                if len(simdoc) < 3 and similarity_matrix[doc_id][ix] > a:
                    if df.iloc[ix]["documents"] in simdoc:
                        continue
                    simdoc.append(df.iloc[ix]["documents"])
                    print('\n')
                    print(f'Document: {df.iloc[ix]["documents"]}')
                    print(f'Cosine Similarity : {similarity_matrix[doc_id][ix]}')
                    print('id : ', ix + 1)
            a -= 0.05

    else:
        df = doc_df(title(news_path))
        print(f'Document: {df.iloc[doc_id]["documents"]}')
        print('\n')
        print('Similar Documents:')
        a = 0.8
        if len(simdoc) >= 3:
            for ix in similar_ix:
                if ix == doc_id:
                    continue
                if len(simdoc) >= 3 and similarity_matrix[doc_id][ix] > 0.8:
                    print('\n')
                    print(f'Document: {df.iloc[ix]["documents"]}')
                    print(f'Cosine Similarity : {similarity_matrix[doc_id][ix]}')
                    print('id : ', ix + 1)
        while (len(simdoc) < 3 or a > 0.5):

            for ix in similar_ix:
                if ix == doc_id:
                    continue
                if len(simdoc) < 3 and similarity_matrix2[doc_id][ix] > a:
                    if df.iloc[ix]["documents"] in simdoc:
                        continue
                    simdoc.append(df.iloc[ix]["documents"])
                    print('\n')
                    print(f'Document: {df.iloc[ix]["documents"]}')
                    print(f'Cosine Similarity : {similarity_matrix[doc_id][ix]}')
                    print('id : ', ix + 1)
            a -= 0.05


# 불용어 처리 함수 여기서 안쓰임
def stopword(path):
    stop_words = []
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            stop_words.append(line)
    return stop_words


# 전처리
def clean(df: pd.DataFrame):
    df['documents_cleaned'] = df.documents.apply(lambda x: " ".join(
        re.sub(r'[^ㄱ-ㅣ가-힣]', ' ', w).lower() for w in x.split()))
    # df['documents_cleaned'] = df.documents.apply(lambda x: " ".join(
    #     re.sub(r'[^ㄱ-ㅣ가-힣]', ' ', w).lower() for w in x.split() if
    #     re.sub(r'[^ㄱ-ㅣ가-힣]', ' ', w).lower() not in stopword(path2)))


df = doc_df(article(news_path))

# 모델 불러오기
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 전처리
clean(df)
print(df.documents_cleaned[0])

vectors = model.encode(df.documents_cleaned)

pairwise_similarities = cosine_similarity(vectors)

# most_similar(0, pairwise_similarities, pairwise_similarities2)
# most_similar(0, pairwise_differences, pairwise_similarities2)
