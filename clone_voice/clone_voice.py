# Load HuBERT for semantic tokens
import os

import torchaudio, torch, numpy as np
from encodec.utils import convert_audio

from bark.generation import _grab_best_device, load_codec_model
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

from clone_voice_config import hubert_model_path, tokenizer_model_name, tokenizer_model_dir, datasets_path

# NotImplementedError: The operator 'aten::_weight_norm_interface' is not currently implemented for the MPS device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = True if device != 'cpu' else False
model = load_codec_model(use_gpu=use_gpu)

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path=hubert_model_path).to(device)

# Load the CustomTokenizer model
tokenizer_model_checkpoint = os.path.join(tokenizer_model_dir, tokenizer_model_name)
tokenizer = CustomTokenizer.load_from_checkpoint(tokenizer_model_checkpoint).to(device)  # Automatically uses the right layers

# Load and pre-process the audio waveform
audio_filepath = os.path.join(datasets_path, 'test', 'audio.wav')    # the audio you want to clone (under 13 seconds)
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()

voice_name = '.'.join(tokenizer_model_name.strip().split('.')[:-1])  # whatever you want the name of the voice to be
output_path = 'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

print('Done!')