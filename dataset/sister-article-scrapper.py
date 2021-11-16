#!pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# !pip install -U scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

# !pip install newspaper3k
from newspaper import Article
from newspaper import Config

# !pip install googlesearch-python
import googlesearch

import pandas as pd
import numpy as np
import datetime
import time
from flask import Flask, jsonify
from flask import Flask, render_template, request

# !pip install ray
import ray

ray.init(num_cpus = 4)


# for shutdow ray.shutdown()

def convertToDate(date):
    try:
        d = pd.to_datetime(date, utc=True)
        return d
    except:
        return date


# Fetching Urls
def getUrls(query):
    # Fetching Articles
    # print('************* FETCHING ARTICLES ***************')

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent

    urlList = []

    for j in googlesearch.search(query, num_results=100, lang="en"):
        urlList.append(j)

    # print('Total urls found: ', len(urlList))

    return urlList


@ray.remote
def getArticles(url):
    try:
        # Here the Article libraray is used which is scraping the articles using urls fetched by googlesearch
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
    except:
        pass

    # Storing the fetched content in dictionary and appending it to a list
    articles = {

        'Title': article.title.strip(),
        'PublishDate': convertToDate(article.publish_date),
        'Author': article.authors,
        'Article': article.text.strip(),
        'Article_Url': url,
        'Article_Length': len(article.text.strip())

    }
    # ARTICLES FETCHED
    # print('************ ARTICLES FETCHED SUCCESSFULLY *****************')
    # print('Total Time taken: ', time.time() - start_time)
    return articles


def similarArticles(data):
    # SENTENCE SIMILARITY
    # print('************ STARTING SENTENCE SIMILARITY ******************')

    # bert-base-nli-mean-tokens maps titles to a 768 dimensional dense vector space
    # and can be used for tasks like clustering or semantic search
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    copy_data = data.copy()
    title = copy_data['Article']

    # Encoding the titles into vectors
    title_vec = model.encode(title)
    # print(title_vec.shape)

    # Matching the vectors similarity
    vectors = cosine_similarity([title_vec[0]], title_vec[1:])

    # Getting indices of vectors which are upto 80% similar
    similar_article_indices = []
    # similar_article_indices = np.where(vectors[0] > 0.6)[0].tolist()
    for list in vectors:
        for i in range(len(list)):
            if list[i] > 0.6:
                similar_article_indices.append(i)

    # print('The Total similar Articles are: ', len(similar_article_indices))

    # Making a new data frame using the similar indices
    raw_data = []
    for j in similar_article_indices:
        raw_data.append(copy_data.iloc[j])
    # raw_data = [copy_data.iloc[j] for j in similar_article_indices]
    similar_title_data = pd.DataFrame(raw_data)
    # print(similar_title_data.isnull().sum())

    # SENTENCE SIMILARITY ENDED
    # print('********** ARTICLES MATCHED SUCCESSFULLY ****************')
    return similar_title_data


def dateFilter(similar_title_data):
    new_data = similar_title_data.dropna(subset=['Title', 'PublishDate', 'Author', 'Article'], how='any')

    # new_data.set_index('PublishDate',drop=False, inplace=True)
    # new_data.index  = similar_title_data.index.tz_convert('US/Eastern')
    # similar_title_data.index  = data.index.tz_localize(None)

    # new_data = similar_title_data.dropna(subset=['Title', 'PublishDate', 'Article'], how='any')

    # Filtering the articles on the basis of 1st article date and articles within two days of that date
    filtered_articles_data = new_data[
        (new_data['PublishDate'] <= pd.to_datetime(new_data['PublishDate'].iloc[0] + datetime.timedelta(days=2))) &
        (new_data['PublishDate'] >= pd.to_datetime(new_data['PublishDate'].iloc[0] - datetime.timedelta(days=2)))
        ]

    # print(filtered_articles_data)
    # filtered_articles_data.to_csv('Filtered_Data.csv')
    # transpose_data = filtered_articles_data.T
    # data_dict = transpose_data.to_dict();

    # FORMATTING DATA
    # print('*********** FORMATTING ENDED SUCCESSFULLY ************ ')
    # print('Total Time taken: ', time.time() - start_time)
    return filtered_articles_data


def scrape(query='The US and Europe have finally reconnected, but theyre moving in different directions on Covid-19'):
    start_time = time.time()

    # Getting Urls
    urls = getUrls(query)

    # Getting Articles List
    article_list = []

    for url in urls:
        article_list.append(getArticles.remote(url))

    raw_data = ray.get(article_list)

    data = pd.DataFrame(raw_data)

    # Doing article similarity
    similar_title_data = similarArticles(data)

    filtered_data = dateFilter(similar_title_data)

    # print(time.time() - start_time)
    return filtered_data


if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()
    scrape()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats("stats_single_vectorized.profile")
