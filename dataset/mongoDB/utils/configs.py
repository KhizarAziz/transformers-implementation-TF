from pymongo import MongoClient

MONGO_CLIENT = MongoClient(
        'mongodb://127.0.0.1:27017')
MONGO_DB = MONGO_CLIENT['CC_db']
MONGO_COL = MONGO_DB['all_news_articles']
