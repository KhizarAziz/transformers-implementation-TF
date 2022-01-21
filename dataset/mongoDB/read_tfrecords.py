import glob
import os
import shutil
import argparse
import tensorflow as tf
 
class Read_tfrecords:
  """
  Reading jsonl files and convert them to tfrecords 
  """

  def __init__(self, input_dir):

    """
    saving neceesary controlable parameters  
    """
    self.input_dir = input_dir
    self.file_type = 'tfrecord'
    self.feature_description = {
        'input_text': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'output_text': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'output_text_score': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'file_path': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    
  def get_files(self, directory):
      """ gets all files of <filetypes> in a directory """
      files = glob.glob(directory+'/*.'+self.file_type)
      return files

  # Create a description of the features for reading tfrecords.
    

  def _parse_function(self, example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, self.feature_description)


  def read_tfrecords(self):
      """
      Read tfrecords files and restore to training/testing format X(example), Y(label)
      """
      tfrecords = self.get_files(self.input_dir)
      print("Total tfrecords : {}".format(len(tfrecords)))
      data = []
      for file_name in tfrecords:
        raw_dataset = tf.data.TFRecordDataset(file_name)
        parsed_dataset = raw_dataset.map(self._parse_function)
        for parsed_record in parsed_dataset:
          input_text  = parsed_record['input_text'].numpy()
          output_text = parsed_record['output_text'].numpy()
          output_text_score = parsed_record['output_text_score'].numpy()
          file_path = parsed_record['file_path'].numpy()
          data.append((input_text,output_text,output_text_score,file_path))

      return data

if __name__ == "__main__":
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str,default='./tf_records/', help="Path to where your tf_rec files are placed")
  args = parser.parse_args()

  input_dir = args.input_dir
  rtfrecords = Read_tfrecords(input_dir)
  data = rtfrecords.read_tfrecords()
  print( "total examples : {} ".format(len(data)))  
