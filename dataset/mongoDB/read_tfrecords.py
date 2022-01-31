import glob
import os
import shutil
import argparse
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Allow memory growth for the GPU
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[5], True)

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

#tf.config.gpu.set_per_process_memory_fraction(0.4)
 
class Read_tfrecords:
  """
  Reading jsonl files and convert them to tfrecords 
  """
  def __init__(self):
    """
    saving neceesary controlable parameters  
    """
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

  def read_tfrecords(self, input_dir):
      """
      Read tfrecords files and restore to training/testing format X(example), Y(label)
      """
      tfrecords = self.get_files(input_dir)
      n = len(tfrecords)
      display = n / 50
      print("Total tfrecords : {}".format(n))
      data = []
      i=0
      for file_name in tfrecords:
        if i%display==0:
          print (i)
        i+=1    
        raw_dataset = tf.data.TFRecordDataset(file_name)
        parsed_dataset = raw_dataset.map(self._parse_function)
        for parsed_record in parsed_dataset:
          input_text  = parsed_record['input_text'].numpy()
          output_text = parsed_record['output_text'].numpy()
          output_text_score = parsed_record['output_text_score'].numpy()
          file_path = parsed_record['file_path'].numpy()
          data.append((input_text,output_text,output_text_score,file_path))
      return data

  def read_tfrecords_generator(self, input_dir):
      """
      Read tfrecords files and restore to training/testing format X(example), Y(label)
      """
      tfrecords = self.get_files(input_dir)
      n = len(tfrecords)
      display = n / 50
      print("Total tfrecords : {}".format(n))
      i=0

      for file_name in tfrecords: 
        if i%display==0:
          print (i)
        i+=1
        raw_dataset = tf.data.TFRecordDataset(file_name)
        parsed_dataset = raw_dataset.map(_parse_function)
        data = []
        for parsed_record in parsed_dataset:
          input_text  = parsed_record['input_text'].numpy().decode('utf-8')
          output_text = parsed_record['output_text'].numpy().decode('utf-8')
          output_text_score = parsed_record['output_text_score'].numpy()
          file_path = parsed_record['file_path'].numpy().decode('utf-8')
          data.append((input_text,output_text,output_text_score,file_path))
        yield data

#Data = []
#for i, data in enumerate(read_tfrecords_generator(tfrecords)):
    #data of single tfrecord
#    print(i, end=',')
#    Data.extend(data) 

    

if __name__ == "__main__":
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str,default='./grouped_articles_new_tfrecords_final/', help="Path to where your tf_rec files are placed")
  args = parser.parse_args()

  input_dir = args.input_dir
  rtfrecords = Read_tfrecords()
  data = rtfrecords.read_tfrecords(input_dir)
  print( "total examples : {} ".format(len(data)))  
