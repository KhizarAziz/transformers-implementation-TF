import requests
from bs4 import BeautifulSoup as bs
import zipfile

domain = 'https://huggingface.co'
url = 'https://huggingface.co/datasets/allenai/c4/tree/main/realnewslike'
FileType = '.gz'

response = requests.get(url)

soup = bs(response.text, 'lxml')

for files in soup.find_all('a', {'class': 'col-span-8 md:col-span-4 lg:col-span-2 truncate flex items-center hover:underline'}):

        file_link = files.get('href').replace('blob', 'resolve')
        file_name = files.text
        print(file_name)
        print(domain + file_link)

        response_url = requests.get(domain + file_link)
        with open(file_name[1:], 'wb') as file:
            file.write(response_url.content)



