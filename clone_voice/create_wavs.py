import os
import random

import numpy
from scipy.io import wavfile

from bark.generation import load_model, SAMPLE_RATE
from bark.api import semantic_to_waveform
from clone_voice_config import wave_path, semantic_path, semantic_dict_path

if not os.path.isdir(semantic_path):
    raise Exception('No \'semantics\' folder, make sure you run create_data.py first!')

if not os.path.isdir(wave_path):
    os.mkdir(wave_path)

print('Loading coarse model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='coarse')

print('Loading fine model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='fine')

for f in os.listdir(semantic_path):
    real_name = '.'.join(f.split('.')[:-1])  # Cut off the extension
    file_name = os.path.join(semantic_path, f)
    out_file = os.path.join(wave_path, f'{real_name}.wav')

    # Don't process files that have already been processed
    if not os.path.isfile(out_file) and os.path.isfile(file_name):
        print(f'Processing {f}')
        wav = semantic_to_waveform(numpy.load(file_name), temp=round(random.uniform(0.6, 0.8), ndigits=2), history_prompt='ja_speaker_0')
        wavfile.write(out_file, SAMPLE_RATE, wav)

print('Done!')
