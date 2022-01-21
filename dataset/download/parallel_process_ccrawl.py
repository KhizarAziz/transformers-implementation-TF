import argparse
import json
import os
import re
from tempfile import TemporaryFile, NamedTemporaryFile
from urllib.parse import urlparse
import boto3
import newspaper
import tldextract
from tqdm import tqdm
from warcio import ArchiveIterator
# from axel import axel
import wget
import glob, pdb
import gzip

with open(os.path.join( 'domain_to_allowed_subdomains.json'), 'r') as f:
    ALLOWED_SUBDOMAINS = json.load(f)

# FOR HANNAH
PROPAGANDA_SUBDOMAINS = {'wnd.com': True, 'infowars.com': True, 'breitbart.com': True, 'dailycaller.com': True,
                         'yournewswire.com': True, 'prageru.com': True, 'newsmax.com': True, 'twitchy.com': True,
                         'dailywire.com': True, 'dailysignal.com': True, 'bigleaguepolitics.com': True,
                         'redstate.com': True, 'townhall.com': True, 'bients.com': True, 'thegatewaypundit.com': True,
                         'nationalreport.net': True, 'naturalnews.com': True, 'prntly.com': True,
                         'worldnewsdailyreport.com': True,
                         'libertywriters.com': True, 'globalresearch.ca': True,
                         }

BANNED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'php', 'css', 'ico', 'xml', 'woff', 'swf', 'jpg', 'svg', 'ttf', 'tif',
                     'bmp', 'js', 'pdf', 'amp', 'rss', 'mp3', 'eot', 'jsp', 'woff2', 'json', 'com', 'axd', 'php3',
                     'bin', 'mp4', 'img', 'xhtml', 'dll', 'm4v', 'vov', 'phtml', 'flv', 'pl', 'jpe', 'otf', 'php\'',
                     'wmv', 'wav', 'xls', 'doc', 'photo', 'gallery', 'bg', 'ece', 'feed', 'xmlhttp', 'video', 'eml',
                     'xnf', 'prt', 'docx', 'file', 'vpx', 'cur', 'data', 'jhtml', 'xlsx', 'map', 'fb', 'webp', 'ppt',
                     'rdf', 'bio', 'exe', 'jar', 'net', 'open', 'ogg', 'wma', '7u', 'res', 'dwr', 'pjpeg', 'gz', 'ajax',
                     'psd', 'zip', 'coffee', 'tabs', 'cls', 'step', 'jp'}

BANNED_STRINGS = ['slideshow.',
                  'slideshowImage', 'associatedcontent.com',
                  '/videoid/', 'sodahead.com', 'b92.net',
                  'isna.ir', 'prnewswire.com', 'slashdot.org', 'suite101.com', 'tv.com', 'news.yahoo.com',
                  '/video/', '/image/', 'bbb.org', 'yle.fi', 'ImageId', 'slideshow_files', '/slideshows/',
                  '/videos/', '/video-', '/videoid/', '/wp-json/', '/search/', 'videoID=', '/portableplayer/',
                  'video.aspx', '/allvideo/', 'width=', 'height=', '/PhotoGallery/', 'ArticleSlideshowServlet',
                  '/storyimage/', '/image.html', '/photos/', '.jpeg', '.jpg', '/em_image', 'maxw=', 'maxh=',
                  '/flashplayers/', '/apps/', '/gallery/', 'photogallery', 'imageViewer', '.jpg', 'img=',
                  '/forums/', '/users/', '/tags/', '/audio/', '/resources/', '/metrics/', '/images/', '/products/',
                  'com.pe', '/agencia/', '/resizer/', '/user?', '/tag/', '/bookmark/', '/plugins/', '/blogs/',
                  '/advertising/', 'blockbuster.co.uk', '/oembed/', '/needlogin', 'type=login', '/mailto/', '/feed',
                  'sendtofriend.aspx', '/ajax/', 'bloggernews.net', '/topics/', 'view_gallery', '/event.asp', '/forum/',
                  '/posts/', '/cgi-bin/', '/member/', 'news_tool_v2.cfm', '/database/', '/Default.aspx',
                  '/Search/', '/Slideshow/', '/slideshow/', '/user/', '/register/', '/donate/', '/calendar/',
                  'send-to-friend',
                  '/enter/', '/photo-gallery/', '/news_email.asp', '/Flash.aspx', '/findlocal/', '/ads/', '/reply/',
                  '/events/', '/picture-gallery/', '/slideshow?', '/Mozilla/', '/sendtoafriend.asp', '/blog/',
                  '/mailStory/', 'admin.asp?', '.ads/', '/used_cars/'
                  ]
BANNED_STRINGS = [re.escape(x) for x in BANNED_STRINGS]
is_banned_regex = re.compile(r'(' + r'|'.join(BANNED_STRINGS) + r')')


def _url_seems_ok(url, domain_to_allowed_subdomains):
    """
    Check if the URL seems ok. if it does then we'll return a tuple of
    CLEAN URL, main domain.
    :param url:
    :return:
    """
    # Long URLs are usually bad
    if len(url) > 200:
        return False

    # FIRST check if the domain is OK
    ext = tldextract.extract(url)
    main_domain = ext.domain + '.' + ext.suffix
    allowed_subdomains = domain_to_allowed_subdomains.get(main_domain, None)
    if allowed_subdomains is None:
        return False

    if isinstance(allowed_subdomains, list) and not ext.subdomain in allowed_subdomains:
        return False

    # Check for banned extensios
    parsed = urlparse(url)
    parsed = parsed._replace(query="", fragment="")
    path_to_use = parsed.path
    file_extension = os.path.splitext(path_to_use)[1]
    if file_extension in BANNED_EXTENSIONS:
        return False

    # If there are two dotcoms then that's probably bad!
    endings = len(re.findall(r'(\.com|\.co\.uk|\.net|\.org)', url))
    if endings > 1:
        return False

    # Check for banned words
    if not (is_banned_regex.search(url) is None):
        return False

    # AT A LATER DATE: we need to check if the URL was banned
    return (parsed.geturl(), main_domain)


def _filter_excessive_newlines(text):
    return re.sub(r'\n\s+', r'\n', text)


class Article(object):
    """ NEWSPAPER VERSION """

    def __init__(self, html):
        self.html = html if html is not None else ""

        self.dummy_article = newspaper.Article(url='', fetch_images=False, verbose=True)
        self.dummy_article.set_html(html)
        self.dummy_article.parse()

        self.text = _filter_excessive_newlines(self.dummy_article.text)
        self.authors = self.dummy_article.authors
        self.authors = [x for x in self.authors if len(x.split(' ')) < 10]
        self.title = self.dummy_article.title

        # sometimes the text started with the title... that's bad
        if self.text.startswith(self.title + '\n'):
            self.text = self.text[len(self.title):].lstrip('\n')

        if self.dummy_article.publish_date and not isinstance(self.dummy_article.publish_date, str):
            try:
                self.publish_date = self.dummy_article.publish_date.date().strftime(
                    "%m-%d-%Y")
            except AttributeError:
                self.publish_date = None
        else:
            self.publish_date = None

        self._extract_summary()

    def _extract_summary(self):
        self.summary = None
        for good2bad in [('og', 'description'), ('twitter', 'description'), ('description',)]:
            curr_dict = self.dummy_article.meta_data
            for key in good2bad[:-1]:
                curr_dict = curr_dict.get(key, {})
            try:    #print(curr_dict)
                #print(type(curr_dict.get(good2bad[-1], '')))
                summary = curr_dict.get(good2bad[-1], '').strip()
            except:
                print(curr_dict.get(good2bad[-1], ''))
                #pass
                summary = str(curr_dict.get(good2bad[-1], '')).strip()

            if len(summary) > 30:
                self.summary = summary
                return
    def num_empty_fields(self):
        num_empty = 0
        for k, v in self.serialize().items():
            if not v:
                num_empty += 1
        return num_empty

    def serialize(self):
        """
        Return simple page object to JSONify and write to file.
        """
        return {
            'meta_lang': self.dummy_article.meta_lang,
            'title': self.title,
            'text': self.text,
            'summary': self.summary,
            'authors': self.authors,
            'publish_date': self.publish_date
        }

    def __repr__(self):
        return str(self.serialize())


def parse_record(record, propaganda=False):
    if record.rec_type != 'response':
        return
    if record.content_type != 'application/http; msgtype=response':
        return

    url_was_ok = _url_seems_ok(record.rec_headers['WARC-Target-URI'],
                               domain_to_allowed_subdomains=PROPAGANDA_SUBDOMAINS if propaganda else ALLOWED_SUBDOMAINS)
    if not url_was_ok:
        return

    url, domain = url_was_ok

    try:
        html = record.content_stream().read().decode('utf-8')
    except UnicodeDecodeError:
        # yield {'status': 'fail', 'url': url, 'reason': 'parse'}
        return

    if not html:
        # yield {'status': 'fail', 'url': url, 'reason': 'parse'}
        return

    try:
        article = Article(html).serialize()
    except ValueError:
        # yield {'status': 'fail', 'url': url, 'reason': 'parse'}
        return

    # Check if is good
    if article['publish_date'] is None:
        # yield {'status': 'fail', 'url': url, 'reason': 'date'}
        return
    if len(article['text']) < 1000:
        # yield {'status': 'fail', 'url': url, 'reason': 'len'}
        return
    if len(article['title']) < 30:
        # yield {'status': 'fail', 'url': url, 'reason': 'title'}
        return

    if article.pop('meta_lang') != 'en':
        # yield {'status': 'fail', 'url': url, 'reason': 'lang'}
        return

    article['status'] = 'success'
    article['url'] = url
    article['domain'] = domain
    article['warc_date'] = record.rec_headers['WARC-Date']
    yield article
   
#####################################################################################################################################


from concurrent.futures import ThreadPoolExecutor
import subprocess
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('-num_of_threads', type=int,
                    default=30,
                    help='number of parallel threads to download and process crawled warc files.')
parser.add_argument('-month_id', type=str,
                    help='CC ID for current month, for which data needs to be fetched... Source: https://commoncrawl.s3.amazonaws.com/cc-index/collections/index.html')
args = parser.parse_args()

#global paramss
month_id = args.month_id # for aprril
max_parallel_threads = args.num_of_threads

# download & extract warc paths
print('DOWNLOADING warc.paths')
wget.download('https://commoncrawl.s3.amazonaws.com/crawl-data/'+month_id+'/warc.paths.gz',month_id+'warc.paths.gz') #download
os.system(f'rm -f {month_id}warc.paths') # remove if any previous
os.system(f'gunzip {month_id}warc.paths.gz') # unzip

if not os.path.exists(month_id):
    os.mkdir(month_id)
    print('\n \nCREATEDd diiir:',month_id)

# open & read all lines from warc file
with open (month_id+'warc.paths', 'r+') as warc_paths:
    warc_lines= warc_paths.read()
    warc_lines = warc_lines.split('\n')
print('\n \nDOWNLOADINGg Filess...... ')
def download_process_file(line):
    file_url = 'https://commoncrawl.s3.amazonaws.com/'+line
    file_name = file_url.split('/')[-1]
    archive_date = line.split('/')[1] # same as month_id e.g CC-MAIN-2021-21
    rest = '_'.join(line.split('/')[2:]) # remaining address     
    out_key = '{}/{}.jsonl'.format(line.split('/')[1], rest)
    if os.path.exists(out_key):
        return f'SKIPPING {file_name}..... Already Downloaded...!'
    try:
        os.system( f'wget {file_url}')
        with gzip.open(file_name, "rb") as r:
            data = r.read()
            with TemporaryFile(mode='w+b') as warctemp:
                warctemp.write(data)
                warctemp.seek(0)
                with open(out_key, 'w') as t:
                    for record in tqdm(ArchiveIterator(warctemp, no_record_parse=False)):
                        for parsed_record in parse_record(record):
                            t.write(json.dumps(parsed_record) + '\n')
        os.system('rm -f ' +file_name)
    except Exception as e:
        print('Exxxcption:: '+file_url.split('/')[-1],e)
        return f"DWNLOADDD FFFAILED: {file_url.split('/')[-1]}"        
    return f'{file_name} DONE!!!'

    
pool = ThreadPoolExecutor(max_workers=max_parallel_threads)
for result in pool.map(download_process_file,warc_lines):
    
    print('\n\nRESULT: ',result)
    
os.system(f'rm -f {month_id}warc.paths') #finally remove warc file fro this month
