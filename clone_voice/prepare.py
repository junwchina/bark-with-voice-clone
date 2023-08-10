import os
import shutil
import numpy
import torchaudio

from hubert.pre_kmeans_hubert import CustomHubert
from clone_voice_config import wave_path, semantic_path, prepared_path, ready_path, hubert_model_path

def prepare():
    """
    Put all the training data in one folder
    :param path: The path to the training data, with 2 subdirectories with zips, "semantic" and "wav", with equal pairs in both directories
    """

    if not os.path.isdir(prepared_path):
        os.mkdir(prepared_path)

    new_offset = 0
    for wave_file in os.listdir(wave_path):
        filename = '.'.join(wave_file.strip().split('.')[:-1])

        # check if file in semantic_path is exists
        semantic_file = os.path.join(semantic_path, f'{filename}.npy')
        if os.path.isfile(semantic_file):
            print(f'Process {filename}')
            shutil.copyfile(os.path.join(wave_path, f'{filename}.wav'), os.path.join(prepared_path, f'{new_offset}_wav.wav'))
            shutil.copyfile(os.path.join(semantic_path, f'{filename}.npy'), os.path.join(prepared_path, f'{new_offset}_semantic.npy'))

            new_offset += 1

    print('All set for prepare')


def prepare2():
    if not os.path.isdir(ready_path):
        os.mkdir(ready_path)

    hubert_model = CustomHubert(checkpoint_path=hubert_model_path)
    wav_string = '_wav.wav'
    sem_string = '_semantic.npy'

    for input_file in os.listdir(prepared_path):
        input_path = os.path.join(prepared_path, input_file)
        if input_file.endswith(wav_string):
            file_num = int(input_file[:-len(wav_string)])
            fname = f'{file_num}_semantic_features.npy'
            output_path = os.path.join(ready_path, fname)
            print('Processing', input_file)
            if os.path.isfile(output_path):
                continue
            wav, sr = torchaudio.load(input_path)

            if wav.shape[0] == 2:  # Stereo to mono if needed
                wav = wav.mean(0, keepdim=True)

            output = hubert_model.forward(wav, input_sample_hz=sr)
            out_array = output.cpu().numpy()
            numpy.save(output_path, out_array)
        elif input_file.endswith(sem_string):
            output_path = os.path.join(ready_path, input_file)
            if os.path.isfile(output_path):
                continue
            shutil.copy(input_path, output_path)
    print('All set! We\'re ready to train!')
