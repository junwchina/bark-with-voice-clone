import os.path

from args import args
from prepare import prepare, prepare2
from clone_voice_config import tokenizer_model_name, tokenizer_model_dir, ready_path
from test_hubert import test_hubert
from hubert.customtokenizer import auto_train

mode = args.mode

if mode == 'prepare':
    prepare()

elif mode == 'prepare2':
    prepare2()

elif mode == 'train':
    if not os.path.exists(tokenizer_model_dir):
        os.mkdir(tokenizer_model_dir)

    model_path = os.path.join(tokenizer_model_dir, tokenizer_model_name)
    auto_train(model_path, ready_path, load_model=model_path, save_epochs=args.train_save_epochs)

elif mode == 'test':
    test_hubert()
