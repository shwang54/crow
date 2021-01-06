# DataHandler for different types of datasets
# Based on http://www.cs.toronto.edu/~nitish/unsupervised_video/

# from util import *
import sys
import numpy as np
import h5py
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# Moving MNIST training dataset construction.
# NOTE:
#   1. dataset size is fixed here.
#   2. Randomly generates moving digits from original mnist.h5 dataset
class BouncingMNISTDataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, train_data, train_label, num_frames, image_size, num_digits, step_length, input_labels):
    self.seq_length_ = num_frames
    self.image_size_ = image_size
    self.image_size_downsample_ = int(self.image_size_ / 2)
    self.num_digits_ = num_digits
    self.step_length_ = step_length
    self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = 14
    self.frame_size_ = self.image_size_ ** 2
    self.input_labels_ = input_labels
    
    self.num_input_labels_ = len(input_labels)
    print(self.num_input_labels_)
    
    for i in range(self.num_input_labels_):
        train_label_find = np.array(np.where(train_label == input_labels[i]))
        if i == 0:
            train_label_ind = train_label_find
        else:
            train_label_ind = np.concatenate((train_label_ind, train_label_find), axis=1)
        print(train_label_ind.shape)
    
    train_label_ind = np.squeeze(train_label_ind)
    print(train_label_ind)

    self.data_ = train_data
    self.labels_ = train_label        # vector of labels
    self.indices_ = train_label_ind   # vector of indices in the dataset
    self.batch_size_ = train_label_ind.shape[0]
    print("self.batch_size_", self.batch_size_)
    self.row_ = 0

    # Global index for consistent test samples
    self.test_indices_ = np.arange(self.data_.shape[0])
    self.test_row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    pass

  def ResetTestIndex(self):
    self.test_row_ = 0


  # -----------------------------------------------------------------------------------------------------------------
  # --------------------- MOVING MNIST WITH CONTROLLED TRAJECTORY
  # -----------------------------------------------------------------------------------------------------------------
  def GetControlledTrajectory(self, my_angle, my_x, my_y, delta_angle, my_speed):
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    # set initial positions of entire batch
    y = np.full(self.batch_size_, my_y)
    if np.size(my_x) > 1:
      x = my_x
    else:
      x = np.full(self.batch_size_, my_x)

    # Choose a random velocity.
    # Use my angle in degrees into theta for the entire batch
    theta = np.full(self.batch_size_, my_angle) / 180.0 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, self.batch_size_))
    start_x = np.zeros((length, self.batch_size_))
    for i in range(length):
      # Take a step along velocity.
      y += v_y * my_speed
      x += v_x * my_speed

      # delta_angle changes deflection ratio. high delta_angle -> low v_x, high v_y, vice versa
      # Bounce off edges.
      for j in range(self.batch_size_):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -(1-delta_angle)*v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -(1-delta_angle)*v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -(1+delta_angle)*v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -(1+delta_angle)*v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  # -----------------------------------------------------------------------------------------------------------------
  # --------------------- GET TRAINING BATCH WITH CONTROLLED TRAJECTORIES
  # -----------------------------------------------------------------------------------------------------------------
  def GetControlledBatch(self, my_angle, my_x, my_y, delta_angle, my_speed, verbose=False):
    
    start_y, start_x = self.GetControlledTrajectory(my_angle=my_angle, my_x=my_x, my_y=my_y, delta_angle=delta_angle, my_speed=my_speed)

    # minibatch data
#     data_downsample = np.zeros((self.batch_size_, self.seq_length_, self.image_size_downsample_, self.image_size_downsample_), dtype=np.float32)
    data = np.zeros((self.batch_size_, self.seq_length_, self.image_size_, self.image_size_), dtype=np.float32)
    data_label = np.zeros(self.batch_size_, dtype=np.float32)
    data_label_onehot = np.zeros((self.batch_size_, self.num_input_labels_), dtype=np.float32)
    
    for j in range(self.batch_size_):
      # get random digit from dataset
      ind = self.indices_[j]
      digit_image = self.data_[ind, :, :]
      digit_image = rescale(digit_image, 1.0 / 2.0, anti_aliasing=False)
    

      # generate video
      for i in range(self.seq_length_):
        top    = start_y[i, j * self.num_digits_]
        left   = start_x[i, j * self.num_digits_]
        bottom = top  + self.digit_size_
        right  = left + self.digit_size_
        data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)
      
      data_label_onehot[j, np.where(self.input_labels_ == self.labels_[ind]) ] = 1
      data_label[j] = self.labels_[ind]                    
        

    return data.reshape(self.batch_size_, -1), data_label, data_label_onehot
