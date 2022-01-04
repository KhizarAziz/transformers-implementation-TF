"""
Created on 02/12/2021

@author: Khizar Aziz
@Email: khizer.awan@gmail.com
"""

import pymongo
import json
from pathlib import Path
import shutil
import tqdm
# def insert_jsonls():


if __name__ == "__main__":
    print('welcom to mongo')
    client = pymongo.MongoClient('mongodb://localhost:27017') # connect to db
    # print(client)

    db = client['CC_db'] # create db with name if doesnt exist else connect to ths db

    collection = db['all_news_articles']
    collection.create_index("text", unique=True)

    jsonls_source_path = Path("data_source")
    dat_backup_dir = Path("data_backup/")
    for jsonl_path in jsonls_source_path.rglob('*.jsonl'): # get all jsonl files from dir and all sub directories of this path.
        with open(jsonl_path) as jsonl_file: # open each jsonl
            print('\n\nReading: ',jsonl_path)
            for line in tqdm.tqdm(jsonl_file):
                try:
                    article = json.loads(line)
                    collection.insert_one(article)
                    #print('SUCCESS:',len(article['text']))
                except Exception as e:
                   # print('FAIL:', len(article['text']))
                    print("Issue: {} || Size: {} ".format(str(e)[:20], len(article['text'])))
                    pass
        r = shutil.move(jsonl_path,dat_backup_dir.joinpath(jsonl_path.name))
        print('All articles done.. source Moved to dir: {} '.format(r))
    print('Converting Publish_dates from dateString to DateObject (ISO) \n\n\n.')
    # convert datestring to dateobject
    collection.update_many({},[{
        "$project": {
            "publish_date": {
                "$dateFromString": {
                    "dateString": "$publish_date",
                    "format": "%m-%d-%Y",
                    "onError": "$publish_date"
                }
            },
            "title": "$title",
            "text": "$text",
            'summary': '$summary',
            'authors': '$authors',
            'url': '$url',
            "status":"$status",
            "domain":"$domain",
            "warc_date":"$warc_date"
        }
    }])
