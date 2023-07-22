import random
import requests

from utils.CJKChars import is_cjk

books = [
    'datasets/ja/ja.text8.txt',  # from https://github.com/Hironsan/ja.text8
]

allowed_chars = '0123456789!@#$%^&*()-_+=\"\':;[]{}/<>,.`~\n\\'


def download_book(book):
    return requests.get(book).content.decode('utf-8')


def read_book(book):
    with open(book, 'r') as f:
        return f.read()


def filter_data(data):
    print('Filtering data')
    return ''.join([char for char in data if char in allowed_chars or is_cjk(char)])


def load_books():
    text_data = []
    print(f'Loading {len(books)} books into ram')
    for book in books:
        text_data.append(load_book(book))
    print('Loaded books')
    return ' '.join(text_data)


def load_book(book):
    print(f'Loading book into ram')
    return filter_data(str(read_book(book)))


def random_split_chunk(data, size=14, spliter=None):
    if spliter is not None:
        data = data.split(spliter)
    index = random.randrange(0, len(data))

    if spliter is not None:
        return spliter.join(data[index:index + size])
    else:
        return data[index:index + size]
