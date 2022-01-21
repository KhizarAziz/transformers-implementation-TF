#Train Tokenizer 
import os
import random
import argparse
import shutil
import glob
from pathlib import Path
from lm_dataformat import Reader
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,processors, trainers)
from tokenizers.normalizers import NFKC
from tqdm import tqdm
import read_tfrecords

class Train_tokenizer:
  def __init__(self, input_dir, vocab_size):

      """
      saving neceesary controlable parameters  
      """
      self.input_dir = input_dir
      self.vocab_size = vocab_size #Hyper-parameter
      self.output_dir = "./out_tokenizer/"
      
      if (os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)):
          shutil.rmtree(self.output_dir)
      os.mkdir(self.output_dir)
      
      self.rtfrecords = read_tfrecords.Read_tfrecords(self.input_dir)

  def train_tokenizer(self):
    """
    Read txt files and train tokenizer and save it json file for encoding / decoding later
    """

    data = self.rtfrecords.read_tfrecords()
    file_path = self.output_dir+"/data.txt"
  
    with open(file_path, "w") as f:
      for example in data:
        input_text , output_text, _ , _ = example
        f.write(input_text.decode('utf-8'))
        f.write("[start] "+output_text.decode('utf-8')+" [end]" )
        f.write("\n\n")
    # assert len(train_data) > 0, 'No data files found'
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()
    # And then train
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>", "<|padding|>"])
    tokenizer.train(trainer, [file_path])

    # And Save it
    tokenizer_path = self.output_dir + "NR-byte-level-bpe.tokenizer.json"
    tokenizer.save(tokenizer_path, pretty=True)

    print(f'\ntokenizer saved at {str(tokenizer_path)}')
    
    return

          
if __name__ == "__main__":
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str, default='./tf_records/', help="Path to where your tf_rec files are placed")
  parser.add_argument("--vocab_size", type=int, default=10000, help="Vocab size is a hyper-parameter, set it accordingly your data size")
  args = parser.parse_args()

  input_dir = args.input_dir
  vocab_size = args.vocab_size
  traintokenizer = Train_tokenizer(input_dir, vocab_size)
  traintokenizer.train_tokenizer()
