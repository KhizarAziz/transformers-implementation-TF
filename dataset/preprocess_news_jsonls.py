"""
Created on 09/11/2021

@author: Khizar Aziz
@Email: khizer.awan@gmail.com
"""


from pathlib import Path
import json
import re
from collections import defaultdict
import random

from bloom_filter import BloomFilter
# this is hella usefl https://krisives.github.io/bloom-calculator/
has_seen_url = BloomFilter(max_elements=14440984416, error_rate=0.1)
has_seen_content_start = BloomFilter(max_elements=14440984416, error_rate=0.10)


TRAIN_PORTION = 0.95
CONTENT_LENGTH = 100

def _get_split(domain):
    """ You could do this by domain, or not"""
    if random.random() < TRAIN_PORTION:
        return 'train'
    return 'val'


def iterate_over_batches(stream, batch_size=64):
    buffer = []
    for x in stream:
        buffer.append(x)
        if len(buffer) >= batch_size:
            yield buffer
            buffer = []
    if len(buffer) > 0:
        yield buffer


def _could_be_author(author):
    author_lower = author.lower().strip()
    if author_lower.startswith(('https', 'www.', 'min read')):
        return False
    if '.com' in author_lower:
        return False
    if author_lower in {'arts', 'politics', 'sports', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
                        'august', 'september', 'october', 'november', 'december'}:
        return False
    return True


def _fix_notfound_authors(article):
    """
    # An extra preprocessing step: if author list is empty and article starts with By then let's fix things.
    :param article:
    :return:
    """
    if len(article['authors']) == 0 and article['text'].startswith('By ') and '\n' in article:
        possible_authors, text = article['text'][3:].split('\n', maxsplit=1)
        if len(possible_authors.split(' ')) < 6:
            article['authors'] = [possible_authors.strip()]
            article['text'] = text.strip()

    article['authors'] = [x for x in article['authors'] if _could_be_author(x)]

    # Those aren't summaries
    if article['summary'] is not None and article['summary'].endswith(('...', '…')):
        article['summary'] = None


# rmove unwanted text
def _fix_photos(article):
    article['text'] += '\n'
    article['text'] = re.sub(
        r'(Facebook Twitter Pinterest |ADVERTISEMENT ADVERTISEMENT|ADVERTISEMENT Thanks for watching! Visit Website)',
        '', article['text'])
    article['text'] = re.sub(r'\nAdvertisement\s+Advertisement\n', '\n', article['text'])

    article['text'] = re.sub(r'\((Photo|Image|Source|Photograph): .{1, 60}\)', '', article['text'])
    article['text'] = re.sub(r'\n(Photo|Image|Source|Photograph): .{1, 60}\n', '\n', article['text'])
    article['text'] = re.sub(r'\nPhoto Published on .{1, 60}\n', '\n', article['text'])

    article['text'] = re.sub(r'\.\s+(Photo|Image): .{1, 60}\n', '.\n', article['text'])
    article['text'] = re.sub(r'\nPicture Courtesy: .{1, 60}\n', '\n', article['text'])
    article['text'] = re.sub(r'\n(\[Related:|RELATED|READ MORE:|PHOTOS:|SEE ALSO:|Also On News|MORE:) .{1, 120}\n',
                             '\n', article['text'])
    article['text'] = re.sub(r'Share this: Facebook\nTwitter\nGoogle\nWhatsApp\nEmail\nCopy\n', '\n', article['text'])

    article['text'] = re.sub(r'\n+', '\n', article['text'])
    # article['text'] = re.sub(r'http.+\b', '', article['text'])
    article['text'].strip()

    # Forbes often has these duplications
    if article['domain'] == 'forbes.com':
        for company_name in ['Apple', 'Microsoft', 'Google', 'Amazon', 'Chase', 'Citigroup', 'Comcast',
                             'Cisco', 'Disney', 'Facebook', 'Intel', 'Netflix', 'Nike', 'Starbucks', 'NVIDIA',
                             'Raytheon', 'Visa', 'Verizon', 'ExxonMobil']:
            article['text'] = article['text'].replace(f'{company_name} {company_name}', f'{company_name}')

def _is_definitely_unique(article):
    # CERTAIN THINGS ALWAYS NEED TO BE BANNED
    if len(re.findall(r'Image \d+ of \d+', article['text'])) > 2:
        return False

    if ' '.join(article['authors']) == 'News Traffic Weather':
        return False

    if article['url'] in has_seen_url:
        return False

    if article['text'][:CONTENT_LENGTH] in has_seen_content_start:
        return False

    has_seen_url.add(article['url'])
    has_seen_content_start.add(article['text'][:CONTENT_LENGTH])
    return True

########################################################################
########################################################################
########################################################################
########################################################################



class Process_News_Data():
    def __init__(self,base_path):
        # self.dataset_name = dataset_name
        self.base_path = Path(base_path)

    # this process reads a json file then process all json strings and return a list of articles
    def article_list_create(self):
        """
        create list of articles against a folder
        it will read all the articles from each file of a directory,
        clean them and append them into a single list

        Args :
            self.base_path --> path of folder having jsonls files
        returns:
                list of unique dictionaries(Articles)

        """
        article_list = []
        for jsonl_file_path in self.base_path.glob('*.jsonl'):
            with open(jsonl_file_path, 'r') as json_file:
                for line in json_file: # reading lines from the file (each line represents each json string because there is a line break at the end of each json)
                    try:
                        article = json.loads(line)
                        # Preprocessing could go here
                        _fix_notfound_authors(article)
                        _fix_photos(article)
                        article_list.append(article)
                    except Exception as e:
                        print("ISSUE: ", json_file, ' Exception: ',e)
                        pass
        return article_list


# define all parameters here
dataset_base_path = 'dummy_jsonl_data/'
dataset_name = 'realnews' # different dataset name means different sequence for loading etc
out_filename = dataset_base_path+'/dummy_output.jsonl'

if __name__ == "__main__":

    dataset_class_ref_object = Process_News_Data(dataset_base_path)
    article_list = dataset_class_ref_object.article_list_create() # convert meta.mat to meta.csv
    print('\n\n ######### All articles loading Successful.. ########### \nTotal Articles: {}'.format(len(article_list)))
    domain2count = defaultdict(int)
    hits = 0
    misses = 0
    with open(out_filename, 'w') as f:
        for article in article_list:
            if _is_definitely_unique(article):
                # working on unique articles only
                domain2count[article['domain']] += 1

                # spliting data into train and validation by adding a label in it
                article['split'] = _get_split(article['domain'])
                if article['split'] != 'ignore':
                    f.write(json.dumps(article) + '\n')
                hits += 1
                if hits % 100000 == 0:
                    print(article, flush=True)

            else:
                misses += 1
                # uncomment the following lines just incase you want
                # to write the duplicates in separate file also
                # else they are not of our use
                """
                article['split'] = _get_split(article['domain'])
                with open(filepath + "/" + name + "_duplicates.jsonl", 'a') as fw:
                    fw.write(json.dumps(article) + '\n')
                """

            if (hits + misses) % 100000 == 0:
                print(f"{hits} hits and {misses} misses", flush=True)
        print(f"{hits} hits and {misses} misses", flush=True)

    # read each jsonl in fastest way
    # de_duplicate/preprocess articles
    # convert to tf_records and save.



