
# coding: utf-8

# In[ ]:




# In[93]:




# In[1]:

import cv2
import tensorflow as tf
import numpy as np
import argparse
import time
import cv
import subprocess as sp
import os
from matplotlib import pyplot as plt


# In[2]:

img = cv2.imread('RIMONE-db-r2/glaucoma/Im256.jpg')
new_img = cv2.resize(img,(28,28))
print np.size(img,0)
print np.size(img,1)
print np.size(new_img,0)
print np.size(new_img,1)
img = cv2.resize(img,(image_size,image_size))
cv2.imshow('image',img)
cv2.imshow('frame',new_img)
rotate_image(img,90)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyWindow('image')


# In[3]:

cv2.destroyAllWindows()



image_size = 100

def rotate(image,degrees):
    
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
 


# In[4]:

glaucomaData = []
glaucomaLabels = []
counter = 0
for i in os.listdir(os.getcwd() + "/RIMONE-db-r2/glaucoma"):
    if i.endswith(".jpg"): 
        
        
        image = cv2.imread("RIMONE-db-r2/glaucoma/" + i)
        
        new_image = cv2.resize(image,(image_size,image_size))
        
        glaucomaData.append(new_image)
        glaucomaLabels.append([0,1])
        
        angle = 90
        while angle < 360:
            
            image = rotate(new_image,angle)
            
            glaucomaData.append(image)
            glaucomaLabels.append([0,1])
            angle = angle + 90
    
        #cv2.imshow('frame',image)
        #print np.size(image,0)
        #print np.size(image,1)
        #print np.size(image,2)
       
        counter = counter + 1
        continue
    else:
        continue
        
        
     


# In[5]:

healthyData = []
healthyLabels = []
counter = 0
for i in os.listdir(os.getcwd() + "/RIMONE-db-r2/normal"):
    if i.endswith(".jpg"): 
        
        
        image = cv2.imread("RIMONE-db-r2/normal/" + i)
        new_image = cv2.resize(image,(image_size,image_size))
        
        healthyData.append(new_image)
        healthyLabels.append([1,0])
        
        angle = 90
        while angle < 360:
            
            image = rotate(new_image,angle)
            
            healthyData.append(image)
            healthyLabels.append([1,0])
            angle = angle + 90

        #cv2.imshow('frame',image)
        #print np.size(image,0)
        #print np.size(image,1)
        #print np.size(image,2)
        counter = counter + 1
        continue
    else:
        continue


# In[ ]:




# In[6]:

print len(glaucomaData)
print len(healthyData)
print np.asarray(healthyData).shape
np.asarray(healthyLabels).shape


# In[7]:

full_data = np.concatenate((glaucomaData,healthyData),0).astype(np.float32)
full_labels = np.concatenate((glaucomaLabels,healthyLabels),0).astype(np.float32)
print full_labels.shape
print full_data.shape


# In[8]:

np.random.seed(133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

full_data,full_labels = randomize(full_data,full_labels)


# In[83]:

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#full_data, full_labels = shuffle_in_unison_inplace(full_data,full_labels)



# In[ ]:


    


# In[9]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[10]:

train_dataset = full_data[0:1600]
train_labels = full_labels[0:1600]
valid_dataset = full_data[1600:1820]
valid_labels = full_labels[1600:1820]


# In[29]:

####################### Lets just be cheap and try it at 28*28.


num_labels = 2
num_channels = 3 # grayscale

batch_size = 10
patch_size = 7
depth = 16
num_hidden = 64

num_steps = 182
# 1820 /5 is 364

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)
  
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  #the reason its /4 is because we've max pooled twice 
    #halved twice (i.e /4)
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size / 4 * image_size / 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  # the way conv2d works is data is the input, layer1weights is the filter, [1,strides,strides,1]
    # is the strides throughtout the dimensions, first and 4th must be the same in 4D,  1. 
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    pool = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding = 'SAME')
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
 # test_prediction = tf.nn.softmax(model(tf_test_dataset))
 


# In[31]:



with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print "Initialized"
  for step in xrange(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 10 == 0):
      print "Minibatch loss at step", step, ":", l
      print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
      print "Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels)
#  print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)



# In[ ]:



