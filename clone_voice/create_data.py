import sys
import os.path
import random
import uuid
import numpy
from bark import text_to_semantic
from bark.generation import load_model
from data import load_book, random_split_chunk

loaded_data = load_book(os.path.join(now_dir, "datasets/ja/ja.text8.txt"))

print('Loading semantics model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')

semantic_path = os.path.join(now_dir, 'datasets/semantic')
semantic_dict = os.path.join(now_dir, 'datasets/semantic_dict.csv')

if not os.path.isdir(semantic_path):
    os.mkdir(semantic_path)

while 1:
    filename = uuid.uuid4().hex + '.npy'
    file_name = os.path.join(semantic_path, filename)
    text = ''
    while not len(text) > 0:
        text = random_split_chunk(loaded_data)  # Obtain a short chunk of text
        text = text.strip()
    print('Generating semantics for text:', text)
    semantics = text_to_semantic(text, temp=round(random.uniform(0.6, 0.8), ndigits=2))
    numpy.save(file_name, semantics)

    # save to file
    with open(semantic_dict, 'a') as f:
        f.write(f'{filename}|{text}\n')
