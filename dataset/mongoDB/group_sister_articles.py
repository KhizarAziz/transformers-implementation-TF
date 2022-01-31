import json
import os
import time
import datetime

from sentence_transformers import SentenceTransformer, util
import FilterData as mongo_agg


def read_jsonl_file(filename):
    print('\nReading Jsonl.....')
    st = time.time()
    f = open(filename, "r", encoding="utf-8")
    objs = f.readlines()
    summaries = set()
    artcile_title_url = dict()

    for obj in objs:
        json_obj = json.loads(obj)
        summaries.add(json_obj['title'] + ' -- ' + json_obj['text'][:100])
        artcile_title_url[json_obj['title'] + ' -- ' + json_obj['text'][:100]] = json_obj

    print(f'\nJsonl read and load time: {time.time() - st}')

    return summaries, artcile_title_url


def read_json_data(objs):
    print('\nReading Json.....')
    st = time.time()
    summaries = set()
    artcile_title_url = dict()

    for json_obj in objs:
        summaries.add(json_obj['title'] + ' -- ' + json_obj['text'][:100])
        artcile_title_url[json_obj['title'] + ' -- ' + json_obj['text'][:100]] = json_obj

    print(f'\nJson data read and load time: {time.time() - st}')

    return summaries, artcile_title_url


def get_embeddings(summaries):
    print('\nEmbedding started.....')
    start = time.time()
    model_name = "all-MiniLM-L6-v2"
    model_path = os.getcwd() + "\\" + model_name
    model = SentenceTransformer(model_path)
    corpus_sentences = list(summaries)
    print("\nEncode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    print(f'\nEmbedding complete: {time.time() - start}')

    return corpus_embeddings


def get_sister_articles(crps_embeddings):
    #print("Start finding sister articles")
    start_time = time.time()

    '''
    Two parameters to tune:
        min_cluster_size: Only consider cluster that have at least 5 elements
        threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    '''
    # clusters = util.community_detection(crps_embeddings, min_community_size=5, threshold=0.75, max_threshold=0.83)
    clusters = util.community_detection_with_score(crps_embeddings, min_community_size=3, threshold=0.75)

    # print("found sister articles after {:.2f} sec".format(time.time() - start_time))

    return clusters


def show_clusters(corps_sentences, clsters, article_obj):
    corpus_sentences = list(corps_sentences)
    # Print for all clusters the top 2 and bottom 2 elements
    for i, cluster in enumerate(clsters):
        print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
        # first_two_clusters = dict(itertools.islice(cluster.items(), 2))
        for sentence_id, score in list(cluster.items())[0:2]:
            print("\tTitle:", "".join(corpus_sentences[sentence_id].split(" -- ")[0]))
            print("\tSummary:", "".join(corpus_sentences[sentence_id].split(" -- ")[1]))
            print("\tUrl:", article_obj[corpus_sentences[sentence_id]])


def save_to_json(corp_sentences, clstrs, output_filename, article_obj, folder):
    corpus_sentences = list(corp_sentences)
    articles_obj = dict()

    for i, clustr in enumerate(clstrs):
        clust_obj = dict()

        for j, (sentence_id, score) in enumerate(clustr.items()):
            sister_article = dict()
            sister_article['title'] = "".join(corpus_sentences[sentence_id].split(" -- ")[0])
            sister_article['url'] = article_obj[corpus_sentences[sentence_id]]['url']
            sister_article['publish_date'] = article_obj[corpus_sentences[sentence_id]]['publish_date'].strftime(
                "%Y-%m-%d")
            try:
                sister_article['summary'] = article_obj[corpus_sentences[sentence_id]]['summary']
            except KeyError:
                pass
            try:
                sister_article['domain'] = article_obj[corpus_sentences[sentence_id]]['domain']
            except KeyError:
                pass
            try:
                sister_article['authors'] = article_obj[corpus_sentences[sentence_id]]['authors']
            except KeyError:
                pass
            try:
                sister_article['split'] = article_obj[corpus_sentences[sentence_id]]['split']
            except KeyError:
                pass
            try:
                sister_article['status'] = article_obj[corpus_sentences[sentence_id]]['status']
            except KeyError:
                pass
            sister_article['text'] = article_obj[corpus_sentences[sentence_id]]['text']
            sister_article['score'] = score
            clust_obj["S" + str(j)] = sister_article
        articles_obj["A" + str(i)] = clust_obj
        # json.dump(articles_obj, f, ensure_ascii=False, indent=4)

        with open(folder + '/' + output_filename, 'w', encoding='utf-8') as f:
            for key, value in articles_obj.items():
                data_dump = dict()
                data_dump[key] = value
                f.write(json.dumps(data_dump))
                f.write('\n')
                # f.write('{%s: %s}\n' % (key, value))


grouped_data = mongo_agg.group_data_by_date()
<<<<<<< HEAD
directory = 'grouped_articles_new'
path = os.path.join(os.getcwd(), directory)
try:
    os.mkdir(path)
except FileExistsError:
    pass


#print(len(grouped_data), "     is the grouped ddata    \n\n\n\n\et_trace()
#pdb.set_trace()
total_groups = len(grouped_data)
for index,date_group in enumerate(zip(grouped_data[0::2], grouped_data[1::2])):
    #print('index:', index)
=======
directory = 'New2days_articles'
path = os.path.join(os.getcwd(), directory)

if not os.path.exists(path):
    os.mkdir(path)
>>>>>>> 51156d97f1ee3feb127a388df218e46e6e094599

for date_group in zip(grouped_data[0::2], grouped_data[1::2]):
    prev_date = date_group[0]
    current_date = date_group[1]
<<<<<<< HEAD
    print('current Date: {} ||| Group# : {}/{}'.format(current_date,index,total_groups))
=======

>>>>>>> 51156d97f1ee3feb127a388df218e46e6e094599
    try:
        docs = mongo_agg.get_data_by_date(prev_date, current_date)
        sentences, article_data = read_json_data(list(docs))
        print(len(article_data))
        if not sentences:
            continue
        embeddings = get_embeddings(sentences)
        sister_articles = get_sister_articles(embeddings)
        if not sister_articles:
            continue
        output_filename = str(prev_date.year) + '-' + str(prev_date.month) + '-' + str(prev_date.day) + '-to-' + str(
            current_date.year) + '-' \
                          + str(current_date.month) + '-' + str(current_date.day) + '-sister-articles.jsonl'
        save_to_json(sentences, sister_articles, output_filename, article_data, directory)

    except TypeError:
        pass
