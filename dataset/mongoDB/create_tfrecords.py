import glob
import jsonlines
import os
import shutil
import argparse
import tensorflow as tf
 
class Create_tfrecords:
  """
  Reading jsonl files and convert them to tfrecords 
  """

  def __init__(self, input_dir, examples_per):

    """
    saving neceesary controlable parameters  
    """
    self.input_dir = input_dir
    self.examples_per = int(examples_per) #examples per tfrecord: # batch_size (input_text -> output_text)
    self.score_limit = [0, 0.95] # sister_article_core_limit
    self.file_type = "jsonl"
    self.output_dir = './tf_records/'

    if (os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)):
      shutil.rmtree(self.output_dir)
    os.mkdir(self.output_dir)
  
  def _bytes_feature(self, value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _float_feature(self, value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _int64_feature(self, value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def serialize_example(self, features):
    """ Creates a tf.train.Example message ready to be written to a file. """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.

    input_text, output_text, score ,file_path = features

    feature = {
        'input_text': self._bytes_feature(input_text.encode('utf-8')),
        'output_text': self._bytes_feature(output_text.encode('utf-8')),
        'output_text_score': self._float_feature(score),
        'file_path': self._bytes_feature(file_path.encode('utf-8'))
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

  def write_to_file(self, writer, features):
      """
      Description   : Writes data to tfrecord file
      input_params  :
                      writer   : tensorflow tfrecord writer pointer
                      features : single data example to store in tfrecord file 
      output_params :  
                      return : nothing but saves a tfrecord file on hard disk 
      """
      writer.write(self.serialize_example(features))

  def data_generator(self, data):
    """Generator function that yield example of a data from list of examples"""
    for features in data:
      yield features
    
  def create_tfrecords(self):
    """ iterates through data files , saving a tfrecords file every <args.num_examples_per_tfrecord> examples. 
    """
    files = self.get_files(self.input_dir)
    print("Total   Jsonl filess: {}".format(len(files)))
    data = self.get_data_xy(files)
    print( "total examples     : {}".format(len(data)))  
    assert (len(data) % self.examples_per) == 0
  
    tokenized_files  = [] 
    tfrecord_counter = 1
    for data_features in self.data_generator(data):
      tokenized_files.append(data_features)
      if len(tokenized_files) == self.examples_per:
        file_name = self.output_dir+str(tfrecord_counter)+'.tfrecord'
        with tf.io.TFRecordWriter(file_name) as writer:
          for f in tokenized_files:
            self.write_to_file(writer, f)
        tfrecord_counter+=1
        tokenized_files = []
    print("Total tfrecords: ".format(tfrecord_counter-1))
    print("output_dir : {} ".format(self.output_dir))

    return 

  def get_files(self, directory):
    """ gets all files of <filetypes> in a directory """
    files = glob.glob(directory+'/*.'+self.file_type)
    return files

  def get_data_xy(self, files):
    """ Read jsonl files and do data labeling in making proper training/testing format X(example), Y(label) """
    processed_jsonl = 0
    data = []
    for i,json_f in enumerate(files):
              with jsonlines.open(json_f) as js_f:
                for article in (js_f.iter()):    
                  article_key = list(article.keys())[0] # sinle json object has single article
                  sister_articles_keys =list(article[article_key].keys())
                  for j in range(len(sister_articles_keys)-1):
                    try:
                      s0 = article[article_key][sister_articles_keys[0]]["title"]
                      sn = article[article_key][sister_articles_keys[j+1]]["title"]  # index error if no sister article
                      sn_score = article[article_key][sister_articles_keys[j+1]]['score']
                      file_path = json_f+'/'+article_key #An
                      file_path+=sister_articles_keys[0] #s0
                      file_path+=sister_articles_keys[j+1] #sn path
                      if (sn_score > self.score_limit[0] and sn_score < self.score_limit[1]):
                        # data.append((s0, sn)) # X,Y
                        data.append((s0, sn, sn_score, file_path))
                    except Exception as err:
                      # pass
                      print('\n\nException error in : {}'.format(json_f))
                      print(err)
                      pass
                      
              processed_jsonl+=1   

    print("processed jsonl files: {}".format(processed_jsonl))        
      
    return data

if __name__ == "__main__":
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str, help="Path to where your jsonl files are located")
  parser.add_argument("--examples_per", type=str, default=50, help="examples per tfrecord")
  args = parser.parse_args()

  input_dir = args.input_dir
  example_per = args.examples_per
  ctfrecords = Create_tfrecords(input_dir, example_per)
  ctfrecords.create_tfrecords()
