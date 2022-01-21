from utils import configs as config
db = config.MONGO_DB
col = config.MONGO_COL



def group_data_by_year_and_month():
    result = col.aggregate(
        [
            {
                "$group":
                    {"_id": {'year': "$year", 'month': "$month"},
                     "total": {"$sum": 1}
                     },
            }
        ]
    )
    year_month = []

    for res in result:
        if res['total'] < 10 or res['_id']['year'] < 1990:
            continue
        year_month.append(res['_id'])
    year_month = sorted(year_month, key=lambda d: d['year'])

    return year_month


def get_data_by_year_month(year, month=1, sort_key='publish_date'):
    docs = col.find({'year': {'$eq': year}, 'month': {'$eq': month}}).sort(sort_key, -1)

    return docs


def group_data_by_date():
    result = col.aggregate(
        [
            {
                "$group":
                    {"_id": {'date': "$publish_date"},
                     "total": {"$sum": 1}
                     },
            }
        ]
    )
    dates = []
    for res in result:
        if res['total'] < 10 or res['_id']['date'].year < 1990:
            continue
        dates.append(res['_id']['date'])

    print('demo dates :',dates[-2:], '\n')
    return sorted(dates)


def get_data_by_date(prev_date, current_date, sort_key='publish_date'):
    docs = col.find({'publish_date': {'$gte': prev_date, '$lte': current_date}}).sort(sort_key, -1)

    return docs
