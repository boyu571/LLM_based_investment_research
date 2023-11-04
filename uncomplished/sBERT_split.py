import json
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity, haversine_distances, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN

news_path = '/content/news.json'

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
    # n = 1
    for doc in tqdm(news, total=len(news)):
        text = doc['content']
        article.append(text)
    return article

# 기사 제목을 반환
def title(path):
    news = koala_news(path)
    title = []
    # n = 1
    for doc in tqdm(news, total=len(news)):
        text = doc['title']
        title.append(text)
    return title

# dataframe화
def doc_df(data):
    documents_df = pd.DataFrame(data, columns=['documents'])
    return documents_df

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
    df['article_cleaned'] = df.article.apply(lambda x: " ".join(
        re.sub(r'[^ㄱ-ㅣ가-힣0-9]', '', w).lower() for w in x.split()))
    df = df[['article_cleaned']]
    df.loc[df['article_cleaned'] == '', 'article_cleaned'] = None
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    # df['documents_cleaned'] = df.documents.apply(lambda x: " ".join(
    #     re.sub(r'[^ㄱ-ㅣ가-힣0-9]', ' ', w).lower() for w in x.split() if
    #     re.sub(r'[^ㄱ-ㅣ가-힣0-9]', ' ', w).lower() not in stopword(path2)))

# 비슷한 문서 찾기
def most_similar(doc_id, similarity_matrix):
    df = doc_df(article(news_path))
    similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    print(f'Document: {df.iloc[doc_id]["documents"]}')
    print('\n')
    for ix in similar_ix:
        if ix == doc_id:
            continue
        if similarity_matrix[doc_id][ix] > 0.85:
            print('Similar Documents:')
            print('\n')
            print(f'Document: {df.iloc[ix]["documents"]}')
            print(f'Cosine Similarity : {similarity_matrix[doc_id][ix]}')
            print('id : ', ix + 1)

# 내용으로 데이터 프레임화
df = doc_df(article(news_path))

# 모델 불러오기
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 제목으로 데이터 프레임화
df_2 = doc_df(title(news_path))

'''
    동적변수 활용해서 하나의 기사를 문장별로 나누고 하나의 기사를 데이터 프레임 한 개로 만듬 그리고 전처리
'''
for l in range(len(df.documents)):
    doc = df['documents'][l].split('.')
    globals()['doc_{}'.format(l)] = pd.DataFrame(doc, columns=['article'])
    globals()['doc_{}'.format(l)] = clean(globals()['doc_{}'.format(l)])


''' 
    문장 임베딩
'''
for i in range(len(df.documents)):
    print(round(i / len(df.documents) * 100, 2), '%')
    globals()['vectors_{}'.format(i)] = model.encode(
        globals()['doc_{}'.format(i)].article_cleaned)  # encode sentences into vectors

''' 
    문장 벡터를 average화 하여 문서 임베딩
'''
for i in range(len(df.documents)):
    globals()['vectors_{}_mean'.format(i)] = globals()['vectors_{}'.format(i)].mean(axis=0)


'''
    흩어져 있는 문서 벡터를 하나로 모음
'''
all_docs_vectors = vectors_0_mean
for i in range(1, len(df.documents)):
    all_docs_vectors = np.vstack((all_docs_vectors, globals()['vectors_{}_mean'.format(i)]))

pairwise_similarities = cosine_similarity(all_docs_vectors)


most_similar(702, pairwise_similarities)

print("##" * 30)
print("DBSCAN Clustering test2")
print("##" * 30)


p = dict()
for a in np.arange(0.01, 1.0, 0.01):
    dbscan = DBSCAN(eps=a, min_samples=2, metric='cosine')
    X = all_docs_vectors
    dbscan_labels = dbscan.fit(X)
    # print(dbscan_labels)
    target = dbscan_labels.fit_predict(X)
    p[a] = max(dbscan_labels.labels_)
max_key = max(p, key=p.get)
print(max_key)

dbscan = DBSCAN(eps=max_key, min_samples=2, metric='cosine')
X = all_docs_vectors
dbscan_labels = dbscan.fit(X)
# print(dbscan_labels)
target = dbscan_labels.fit_predict(X)

cluster_dict = {i: [] for i in range(max(dbscan_labels.labels_) + 1)}

# print(dbscan_labels.labels_)
# print(cluster_dict)



for i, label in zip(range(len(df.documents)), dbscan_labels.labels_):
    if label != -1:
        cluster_dict[label].append(i)
for label, lst in cluster_dict.items():
    if label != -1:
        print(f"Cluster {label}")
        for x in lst:
            print(x)
        print("--" * 30)
    print("##" * 20)