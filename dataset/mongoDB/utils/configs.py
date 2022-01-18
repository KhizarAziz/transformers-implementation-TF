from pymongo import MongoClient

MONGO_CLIENT = MongoClient(
    'mongodb://localhost:27017')
MONGO_DB = MONGO_CLIENT['CC_db']
MONGO_COL = MONGO_CLIENT['all_news_articles']