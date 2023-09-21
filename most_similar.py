def most_similar(doc_id, similarity_matrix, similarity_matrix2, matrix):
    df = doc_df(article(news_path))
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_id])
    simdoc = []
    for ix in similar_ix:
        if ix == doc_id:
            continue
        if similarity_matrix[doc_id][ix] > 0.8:
            simdoc.append(df.iloc[ix]["documents"])
    if len(simdoc) < 6:
        print(f'Document: {df.iloc[doc_id]["documents"]}')
        print('\n')
        print('Similar Documents:')
        for ix in similar_ix:
            if ix == doc_id:
                continue
            if similarity_matrix[doc_id][ix] > 0.8:
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 1 and similarity_matrix[doc_id][ix] > 0.75:
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 3 and similarity_matrix[doc_id][ix] > 0.7:
                if df.iloc[ix]["documents"] in simdoc:
                    continue
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 3 and similarity_matrix[doc_id][ix] > 0.65:
                if df.iloc[ix]["documents"] in simdoc:
                    continue
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
                print('id : ', ix + 1)
    else:
        df = doc_df(title(news_path))
        print(f'Document: {df.iloc[doc_id]["documents"]}')
        print('\n')
        print('Similar Documents:')
        for ix in similar_ix:
            if ix == doc_id:
                continue
            if similarity_matrix2[doc_id][ix] > 0.8:
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix2[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 1 and similarity_matrix2[doc_id][ix] > 0.75:
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix2[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 3 and similarity_matrix2[doc_id][ix] > 0.7:
                if df.iloc[ix]["documents"] in simdoc:
                    continue
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix2[doc_id][ix]}')
                print('id : ', ix + 1)

        for ix in similar_ix:
            if ix == doc_id:
                continue
            if len(simdoc) < 3 and similarity_matrix2[doc_id][ix] > 0.65:
                if df.iloc[ix]["documents"] in simdoc:
                    continue
                simdoc.append(df.iloc[ix]["documents"])
                print('\n')
                print(f'Document: {df.iloc[ix]["documents"]}')
                print(f'{matrix} : {similarity_matrix2[doc_id][ix]}')
                print('id : ', ix + 1)