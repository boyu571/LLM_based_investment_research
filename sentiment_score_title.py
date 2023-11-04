import os
import time

import openai
import pandas as pd

# openai.api_key = 
# openai.api_base = 
# openai.api_type = 
# openai.api_version = 

deployment_name = "gpt-35-turbo-16k"


data = pd.read_csv('/workspace/llm/sentiment/code/sim_del_naver_news_data_v3.csv')
data = data.iloc

def get_completion_16k(prompt, model=deployment_name, max_tokens=16384):
    # 프롬프트를 토큰으로 변환하여 최대 허용 토큰 수를 초과하는지 확인합니다.
    # 여기서는 `openai.Tokenizer.encode`를 사용하여 정확한 토큰 수를 계산합니다.
    prompt_tokens = openai.Tokenizer.encode(prompt)
    
    if len(prompt_tokens) > max_tokens:
        allowed_text = openai.Tokenizer.decode(prompt_tokens[:max_tokens])
    else:
        allowed_text = prompt

    messages = [{"role": "user", "content": allowed_text}]
    try: 
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            model=model,
            messages=messages,
            temperature=0,
        )
        result = response.choices[0].message["content"]
    except Exception as e:
        # 오류 메시지를 출력하거나 로깅할 수 있습니다.
        print(f"An error occurred: {e}")
        result = 9999  

def get_completion_with_delay(x):
    result = get_completion_16k(x)
    time.sleep(2)
    return result


prompt2 = "이전 지시사항은 모두 무시하십시오. 당신은 금융 전문가로 가정하며 금융자산 추천 경험이 있습니다. 만약 기사의 제목이 다음날 한국에서 거래되는 암호화폐의 가격에 좋은 소식이 확실하다면 1에 가까운 수를, 나쁜 소식이 확실하다면 –1에 가까운 수, 불확실하거나 암호화폐 가격에 대한 정보가 부족한 경우 0에 가까운 수로 답변해주세요. 다시 말해, 뉴스 기사 제목을 읽고 다음날 암호화폐 가격에 대해 좋은 소식인지, 아니면 나쁜 소식인지, 혹은 불확실한지를 –1과 1 사이의 실수로만 답하세요. 기사 내에 여러 개의 기사나 내용이 있다면 종합적으로 고려하여 여러개의 숫자가 아닌 하나의 숫자로만 답변하세요. 설명이나 추가적인 텍스트 없이 숫자만 답변하세요."

# 결과를 저장할 빈 데이터프레임을 초기화합니다.
result2 = pd.DataFrame()

# data의 모든 행을 100행씩 처리합니다.
for i in range(0, len(data), 100):
    subset = data.iloc[i:i+100]
    subset['score_title'] = (prompt2 + '제목:' + subset['title']).apply(get_completion_with_delay)
    result2 = pd.concat([result2, subset])

result2.reset_index(drop=True, inplace=True)
result2.to_csv('/workspace/llm/sentiment/code/result2_22208.csv', encoding='utf-8-sig')