import os.path

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

from clone_voice_config import tokenizer_model_name

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
こんにちは、初めまして
"""

voice_name = '.'.join(tokenizer_model_name.strip().split('.')[:-1])  # whatever you want the name of the voice to be
audio_array = generate_audio(text_prompt, history_prompt=voice_name)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)

print('Done')