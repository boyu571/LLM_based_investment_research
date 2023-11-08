import os
import time

import openai
import pandas as pd
import tiktoken
# max token 처리를 위해 tiktoken 라이브러리 설치 필요

# openai.api_key =
# openai.api_base = 
# openai.api_type = 
# openai.api_version = 

deployment_name = "gpt-3.5-turbo-1106"
# 1106에 새로 업데이트 된 gpt-3.5-turbo-1106는 16k까지 처리 가능합니다.

data = pd.read_csv('/workspace/llm/sentiment/code/eng_ver1_part1.csv')

# 여기서부터 get_completion 함수 수정했습니다.(max token)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
max_tokens = 16300

def get_completion(prompt, model=deployment_name):
    try:
        # 텍스트를 토큰으로 변환합니다.
        prompt_tokens = encoding.encode(prompt)
        
        # 토큰의 길이를 확인하고, max_tokens를 초과하지 않도록 조정합니다.
        if len(prompt_tokens) > max_tokens:
            prompt_tokens = prompt_tokens[:max_tokens]
            # 인코딩된 토큰을 다시 텍스트로 변환합니다.
            # tiktoken 라이브러리에는 decode 메서드가 없을 수 있으므로, 이 부분은 필요에 따라 조정해야 합니다.
            allowed_text = encoding.decode(prompt_tokens)
        else:
            allowed_text = prompt

        # 수정된 텍스트로 API 요청을 보냅니다.
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            model=model,
            messages=[{"role": "user", "content": allowed_text}],
            temperature=0,
        )
        result = response.choices[0].message["content"]
    except Exception as e:
        print(f"An error occurred: {e}")
        result = 9999
    return result

def get_completion_with_delay(x):
    result = get_completion(x)
    time.sleep(2)
    return result

# (수정) prompt
prompt1 = "Forget all previous instructions. Pretend you are a financial expert with experience in recommending financial assets. If the headline and content of the article are certain to be good news for the price of cryptocurrencies traded the next day, answer with a number close to 1. If it is certain to be bad news, answer with a number close to -1. If it is uncertain or there is insufficient information about the impact on cryptocurrency prices, answer with a number close to 0. In other words, after reading the news article headline and content, respond with a real number between -1 and 1, indicating whether it's good news, bad news, or uncertain for the cryptocurrency price the next day. If there are multiple articles or contents in the article, consider them comprehensively and respond with only one number instead of multiple numbers. Answer with the number only, without explanation or additional text."

# 결과를 저장할 빈 데이터프레임을 초기화합니다.
result1 = pd.DataFrame()

# (수정) subset['content_full'] / data의 모든 행을 100행씩 처리합니다.
for i in range(0, len(data), 100):
    subset = data.iloc[i:i+100]
    subset['score'] = (prompt1 + '제목:' + subset['title'] + '내용:' + subset['content_full']).apply(get_completion_with_delay)
    result1 = pd.concat([result1, subset])
    # result1.to_csv(f'/workspace/llm/sentiment/code/result1_{i+100}.csv', encoding='utf-8-sig')
    
result1.reset_index(drop=True, inplace=True)
result1.to_csv('/workspace/llm/sentiment/code/result1.csv', encoding='utf-8-sig')