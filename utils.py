"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import math
import tensorflow as tf

import mrc
FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, config):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  if config.scale:
      scale = config.scale
  else:
      scale = 3
  
  stride = config.stride

  image = imread(path, is_grayscale=True)
  
  label_ = modcrop(image, scale)
  
  h, w, d = label_.shape
  
  pads = abs(config.image_size-config.label_size)
  
  # Calculate the number of times stride can go through the size - padding
  nh = int(math.ceil((h-pads)/float(stride))) + 1
  nw = int(math.ceil((w-pads)/float(stride))) + 1
  nd = int(math.ceil((d-pads)/float(stride))) + 1

  # 
  h_hat = (stride * nh) + pads
  w_hat = (stride * nw) + pads
  d_hat = (stride * nd) + pads
  
  # Padding needed to make the original image size bigger
  h_pad = int(math.ceil(abs(h-h_hat)/2.0))
  w_pad = int(math.ceil(abs(w-w_hat)/2.0))
  d_pad = int(math.ceil(abs(d-d_hat)/2.0))

  # Pad the original image using 'edge'
  label_ = np.pad(label_, ((h_pad, h_pad), (w_pad, w_pad), (d_pad, d_pad)), 'edge')
 
  # Scaling using bicubic
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
  print("Interpolation is done")
  #input_ = scipy.misc.imresize(label_, (1./scale), interp='bicubic')
  #input_ = scipy.misc.imresize(input_, (scale/1.), interp='bicubic')
  
  #print(input_.shape, label_.shape) 
  
  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  filetype = 'mrc'
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "mrc_files")
    print(data_dir)
    data = glob.glob(os.path.join(data_dir, "*.%s" % (filetype)))
  else:
   data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "mrc_files")
   data = glob.glob(os.path.join(data_dir, "*.%s" % (filetype)))

  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  return mrc.readMRC(path)
  #if is_grayscale:
    #return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  #else:
    #return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 4:
    h, w, d, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    d = d - np.mod(d, scale)
    image = image[0:h, 0:w, 0:d, :]
  elif len(image.shape) == 3:
    h, w, d = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    d = d - np.mod(d, scale)
    image = image[0:h, 0:w, 0:d]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  print(sess)
  print(config)
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train")
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) / 2 # 6

  if config.is_train:
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], config)

      #image_path = os.path.join(os.getcwd(), 'processed_images_inputs')
      #image_path = os.path.join(image_path, data[i].split('/')[-1])
      #imsave(input_, image_path)
      #data_file_inputs = os.path.join(os.path.join(os.getcwd(), 'processed_images_inputs'), data[i].split('/')[-1].split('.')[0])
      #data_file_inputs = os.path.join(data_file_inputs, 'sub_images')
      
      #if not os.path.exists(data_file_inputs):
         # os.makedirs(data_file_inputs)
      
      #image_path = os.path.join(os.getcwd(), 'processed_images_labels')
      #image_path = os.path.join(image_path, data[i].split('/')[-1])
      #imsave(label_, image_path)
      #data_file_labels = os.path.join(os.path.join(os.getcwd(), 'processed_images_labels'), data[i].split('/')[-1].split('.')[0])
      #data_file_labels = os.path.join(data_file_labels, 'sub_images')
     
      #if not os.path.exists(data_file_labels):
         # os.makedirs(data_file_labels)
      if len(input_.shape) == 4:
        h, w, d, _ = input_.shape
      elif len(input_.shape) == 3:
        h, w, d = input_.shape
      else:
        h, w = input_.shape

      for z in range(0, d-config.image_size+1, config.stride): 
        for x in range(0, h-config.image_size+1, config.stride):
            for y in range(0, w-config.image_size+1, config.stride):
                sub_input = input_[x:x+config.image_size, y:y+config.image_size, z:z+config.image_size] # [33 x 33 x 33]
                sub_label = label_[int(x+padding):int(x+padding+config.label_size), int(y+padding):int(y+padding+config.label_size), int(z+padding):int(z+padding+config.label_size)] # [21 x 21 x 21] 
          
          #inputs_path = os.path.join(data_file_inputs, '%s-%s.bmp' % (x, y)) 
          #labels_path = os.path.join(data_file_labels, '%s-%s.bmp' % (x, y))
         
          #imsave(sub_input, inputs_path) 
          #imsave(sub_label, labels_path)
          # Make channel value
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.image_size, 1])  
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.label_size, 1])
        
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
  else:
    input_, label_ = preprocess(data[0], config)
    
    if len(input_.shape) == 4:
      h, w, d, _ = input_.shape
    elif len(input_.shape) == 3:
      h, w, d = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = nz = 0
    for z in range(0, d-config.image_size+1, config.stride):
      nz +=1; nx = 0;
      for x in range(0, h-config.image_size+1, config.stride):
        nx += 1; ny = 0; 
        for y in range(0, w-config.image_size+1, config.stride):
          ny += 1;
          sub_input = input_[x:x+config.image_size, y:y+config.image_size, z:z+config.image_size] # [33 x 33]
          sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size, z+padding:z+padding+config.label_size] # [21 x 21 x 21]
        
          sub_input = sub_input.reshape([config.image_size, config.image_size, config.image_size, 1])  
          sub_label = sub_label.reshape([config.label_size, config.label_size, config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 21, 1]
  #print (arrdata.shape, arrlabel.shape)
  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
    return nx, ny, nz
    
def imsave(image, path):
  return mrc.writeMRC(path, image)
  #return scipy.misc.imsave(path, image)

def merge(images, size):
  h, w, d = images.shape[1], images.shape[2], images.shape[3]
  print (images.shape)
  print ("Dimensions: %s, %s, %s" % (h, w, d))
  print ("nx, ny, nz: %s, %s, %s" % (size[0], size[1], size[2]))
  img = np.zeros((h*size[0], w*size[1], d*size[2], 1))
  print(images[0].shape)
  print(img.shape)
  #for z in xrange(d-1):
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = (idx // (size[1])) % size[1]
    k = idx // (size[1] * size[1])
    #print("idx, size[1]: %s, %s" % (idx, size[1]))
    #print("i, j: %s, %s, %s" % (i, j, k))
    print (img[j*h:j*h+h, i*w:i*w+w, k*d:k*d+d, :].shape, image.shape)
    img[j*h:j*h+h, i*w:i*w+w, k*d:k*d+d, :] = image 
  return img
