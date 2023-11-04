def most_similar(doc_id, similarity_matrix, matrix):
    df = doc_df(article(news_path))
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_id])
    print(f'Document: {df.iloc[doc_id]["documents"]}')
    print('\n')
    for ix in similar_ix:
        if ix == doc_id:
            continue
        if similarity_matrix[doc_id][ix] > 0.8:
            print('Similar Documents:')
            print('\n')
            print (f'Document: {df.iloc[ix]["documents"]}')
            print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
            print('id : ', ix + 1)