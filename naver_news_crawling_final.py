import requests
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
from tqdm import tqdm

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

def makeUrl(search, start_date, end_date, start_pg=1, end_pg=2):
    date_part = f"&pd=3&ds={start_date}&de={end_date}"
    base_url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}"
    return [base_url.format(search) + date_part + f"&sort=2&start={(i - 1) * 10 + 1}" for i in range(start_pg, end_pg + 1)]

def get_naver_articles(urls):
    articles, titles, dates, press = [], [], [], []
    date_pattern = re.compile(r"\d{4}\.\d{2}\.\d{2}")
    
    for url in urls:
        html = requests.get(url, headers=headers)
        soup = BeautifulSoup(html.text, "html.parser")

        article_links = [link['href'] for link in soup.select(".info_group > a.info")]
        article_titles = [title.text for title in soup.select(".news_tit")]
        article_infos = [info.text for info in soup.select(".info")]

        # Extract article links based on the 'news.naver.com' domain
        new_urls = [article_links[i] for i in range(len(article_links)) if 'news.naver.com' in article_links[i] or (i < len(article_links) - 1 and 'news.naver.com' not in article_links[i+1])]

        current_dates = [date_pattern.search(info).group(0) for info in article_infos if date_pattern.search(info)]
        current_press = [article_infos[i-1] for i, info in enumerate(article_infos) if date_pattern.search(info) and i > 0]

        articles.extend(new_urls)
        titles.extend(article_titles)
        dates.extend(current_dates)
        press.extend(current_press)

        time.sleep(1)
        
    return articles, titles, dates, press

def get_article_contents(article_urls):
    contents, article_time = [], []

    for url in tqdm(article_urls):
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
        content = soup.select_one("#dic_area, #articeBody")
        contents.append(re.sub('<[^>]*>', '', content.text).replace("flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}", ''))
        
        date = soup.select_one(".media_end_head_info_datestamp > div > span") or soup.select_one("#content > div.end_ct > div > div.article_info > span > em")
        article_time.append(re.sub('<[^>]*>', '', date.text))

    return contents, article_time

def main():
    search = input("검색할 키워드를 입력해주세요: ")
    start_date = input("\n크롤링할 시작 날짜를 입력해주세요. ex)2022.01.01: ")
    end_date = input("\n크롤링할 종료 날짜를 입력해주세요. ex)2022.12.31: ")

    urls = makeUrl(search, start_date, end_date)
    article_urls, article_titles, article_dates, article_press = get_naver_articles(urls)
    contents, article_time = get_article_contents(article_urls)

    df = pd.DataFrame({'date': article_dates, 'time': article_time, 'title': article_titles, 'content': contents, 'press': article_press, 'link': article_urls})
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    filename = f"{search}_{start_date.replace('.', '')}_{end_date.replace('.', '')}.csv"
    df.to_csv(filename, encoding='utf-8-sig', index=False)
    
    return df

if __name__ == "__main__":
    result_df = main()
    print(result_df)
