from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import errno
import os
import re

import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # pylint: disable=line-too-long


# Control downsampling: how many scalar data do we keep for each run/tag
# combination?
SIZE_GUIDANCE = {'scalars': 100000}


def extract_scalars(multiplexer, run, tag):
  """Extract tabular data from the scalars at a given run and tag.

  The result is a list of 3-tuples (wall_time, step, value).
  """
  tensor_events = multiplexer.Tensors(run, tag)
  return [
      (event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item())
      for event in tensor_events
  ]


def create_multiplexer(logdir):
  multiplexer = event_multiplexer.EventMultiplexer(
      tensor_size_guidance=SIZE_GUIDANCE)
  multiplexer.AddRunsFromDirectory(logdir)
  multiplexer.Reload()
  return multiplexer


def export_scalars(multiplexer, run, tag, filepath, write_headers=True):
  data = extract_scalars(multiplexer, run, tag)
  with open(filepath, 'w') as outfile:
    writer = csv.writer(outfile)
    if write_headers:
      writer.writerow(('wall_time', 'step', 'value'))
    for row in data:
      writer.writerow(row)


NON_ALPHABETIC = re.compile('[^A-Za-z0-9_]')

def munge_filename(name):
  """Remove characters that might not be safe in a filename."""
  return NON_ALPHABETIC.sub('_', name)


def mkdir_p(directory):
  try:
    os.makedirs(directory)
  except OSError as e:
    if not (e.errno == errno.EEXIST and os.path.isdir(directory)):
      raise


def main():
  tag_name = ('/total r')
  logdir = 'logs/checkpoints'
  output_dir = './csv_output'
  mkdir_p(output_dir)


  print("Loading data...")
  for root, subdir, files in os.walk(logdir):
    if files:
      if "use_seperate_networks=True" not in root:
        root_s = root
        runname = "/".join( (root_s.split('/'))[2:] )
        root = root.split('/')
        env_name = next((t.split('='))[-1] for t in root if "env_name" in t)
        multiplexer = create_multiplexer(logdir)
        tag = "%s%s"%(env_name,tag_name)
        print("extracting tag: %s" % tag)
        output_filename = munge_filename(tag)
        output_filepath = os.path.join(output_dir, output_filename+'.csv')
        print(
            "Exporting (run=%r, tag=%r) to %r..."
            % (runname, env_name+tag_name, output_filepath))
        export_scalars(multiplexer, runname, tag, output_filepath)

  print("Done.")

def test():
  tag_name = ('/total r')
  logdir = './logs/checkpoints'
  output_dir = './csv_output'
  mkdir_p(output_dir)


  print("Loading data...")
  for root, subdir, files in os.walk(logdir):
    if files:
      if "use_seperate_networks=True" not in root:
        root_s = root
        root = root.split('/')
        env_name = next((t.split('='))[-1] for t in root if "env_name" in t)
        output_filename = '%s-%s' % (
            munge_filename(env_name), munge_filename(tag_name))
        output_filepath = os.path.join(output_dir, output_filename+'.csv')
        print(
            "Exporting (run=%r, tag=%r) to %r..."
            % (env_name, tag_name, output_filepath))
        print(root_s)
        print(env_name+tag_name)
  print("Done.")

def one_at_a_time():
  tag_name = ('/total r')
  logdir = './'
  output_dir = '/home/bing/Dropbox/10_semester/csv_output'
  mkdir_p(output_dir)


  print("Loading data...")

  env_name = 'Reacher-v1'
  multiplexer = create_multiplexer(logdir)
  tag = "%s%s"%(env_name,tag_name)
  print("extracting tag: %s" % tag)
  output_filename = munge_filename(tag)
  output_filepath = os.path.join(output_dir, output_filename+'.csv')
  print(
      "Exporting (run=%r, tag=%r) to %r..."
      % ('./', env_name+tag_name, output_filepath))
  export_scalars(multiplexer, './', tag, output_filepath)
  print("Done.")

if __name__ == '__main__':
  main()
  #test()
  #one_at_a_time()