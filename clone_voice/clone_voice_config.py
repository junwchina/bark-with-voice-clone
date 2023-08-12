import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(now_dir, "datasets")
ja_book_path = os.path.join(datasets_path, "ja/ja.text8.txt")
semantic_path = os.path.join(datasets_path, 'semantic')
wave_path = os.path.join(datasets_path, 'wav')
semantic_dict_path = os.path.join(datasets_path, 'semantic_dict.csv')
prepared_path = os.path.join(datasets_path, 'prepared')
ready_path = os.path.join(datasets_path, 'ready')
hubert_model_path = os.path.join(now_dir, 'models/hubert_base_ls960.pt')
tokenizer_model_name = "ja_tokenizer.pth"
tokenizer_model_dir = os.path.join(datasets_path, 'models')