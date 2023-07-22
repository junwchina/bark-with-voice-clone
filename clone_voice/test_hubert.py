import os

import numpy
import torchaudio

from clone_voice_config import hubert_model_path, tokenizer_model_dir, tokenizer_model_name, datasets_path
from hubert.customtokenizer import CustomTokenizer
from hubert.pre_kmeans_hubert import CustomHubert


def test_hubert(model_checkpoint=None):
    hubert_model = CustomHubert(checkpoint_path=hubert_model_path)

    if model_checkpoint is None:
        model_checkpoint = os.path.join(tokenizer_model_dir, tokenizer_model_name)

    customtokenizer = CustomTokenizer.load_from_checkpoint(model_checkpoint)

    wav, sr = torchaudio.load(os.path.join(datasets_path, 'test', 'wav.wav'))
    original = numpy.load(os.path.join(datasets_path, 'test', 'semantic.npy'))

    out = hubert_model.forward(wav, input_sample_hz=sr)
    out_tokenized = customtokenizer.get_token(out)

    # print(out.shape, out_tokenized.shape)
    print(original[:-1], out_tokenized)
    numpy.save(os.path.join(datasets_path, 'test', 'gen_semantic.npy'), out_tokenized)
