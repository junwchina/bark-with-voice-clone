import os
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import AudioDataStream
from clone_voice_config import datasets_path


def get_azure_speech_synthesizer(voice_name):
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                           region=os.environ.get('SPEECH_REGION'))
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name = voice_name

    # Creates an audio configuration that points to an audio file.

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    return speech_synthesizer


def synthesis_text(text, voice_name='ja-JP-NanamiNeural'):
    speech_synthesizer = get_azure_speech_synthesizer(voice_name)
    result = speech_synthesizer.speak_text_async(text).get()
    stream = AudioDataStream(result)
    return stream


if __name__ == '__main__':
    synthesis_text("夜になり、線状降水帯による非常に激しい雨が降り続いているとして、気象庁は午後9時に宮崎県と熊本県に顕著な大雨に関する情報を発表しました。").save_to_wav_file(os.path.join(datasets_path, "test/japanese_prompt.wav"))