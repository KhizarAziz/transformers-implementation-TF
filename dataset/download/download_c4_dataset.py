import requests

for i in range(0,512):
    url = (f'https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-train.{i:05}-of-00512.json.gz')
    filename = f'{i}c4-train.{i:05}-of-00512.json.gz'
    print(filename)
    print(url)

    response_url = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response_url.content)
