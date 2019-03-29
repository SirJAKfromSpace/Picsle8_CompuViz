#required libraries 
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 
import pandas as pd 
import os


#####
# @Desc: Returns a dictioanry that contains one hot encoded arrays for each label. 
#        The keys of the dictionary are the string labels.
# @returns: dict <python dictionary>
#####
def one_hot_encoded_label_dict():

    dict = {"A" : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "B" : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "C" : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "D" : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "E" : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "F" : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "G" : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "H" : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "I" : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "J" : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "K" : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "L" : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "M" : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "N" : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  
            "O" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "P" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "Q" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], 
            "R" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], 
            "S" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 
            "T" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 
            "U" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 
            "V" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 
            "W" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 
            "X" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], 
            "Y" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 
            "Z" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 
            "SPACE" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 
            "NOTHING" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 
            "DEL" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
            }

    return dict


#####
# @Desc: Loads the image dataset into batches tensors that can be read by tensorflow.
#       
# @params: dataset_dir <str> the dataset directory
#          image_type <str> the file extention of the images to be used
#          batch_size <int> the size of each tensorbatch
#   
# @returns: X --> values of the tensor batches 
#           Y --> labels of the tensor batches
# @References:
#               1. https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
#               2. https://youtu.be/umGJ30-15_A
#               3. http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
#               4. https://www.tensorflow.org/guide/datasets
#####
def prp_dataset(dataset_dir, image_type = '.jpg', batch_size = 100, sample_size=1000, test_perc=0.10):

    imagepaths = [] #Will store the directory of each image 
    labels = []     #Will store a list of one hot encoded labels

    test_imagepaths = []
    test_labels = [] 

    one_hot_dict = one_hot_encoded_label_dict()

    folders = sorted(os.walk(dataset_dir).__next__()[1])

    #traversing the outer Folder
    for folder_name in folders:

        count = 0

        folder_label = one_hot_dict[folder_name.upper()] #The one hot encoded label for the folder's contents 

        inner_folder_dir = os.path.join(dataset_dir, folder_name)
        
        image_files = os.walk(inner_folder_dir).__next__()[2] # [1] for folders, [2] for files, [0] not sure

        #print(len(image_files))

        #traversing the inner directory and looping through every image
        for image_file in image_files:
            
            if image_file.endswith(image_type) and count < (sample_size - test_perc*sample_size): #ensuring only intended imagetype is stored
                imagepaths.append(os.path.join(inner_folder_dir, image_file))
                labels.append(folder_label)
                count+=1
            elif count < sample_size:
                test_imagepaths.append(os.path.join(inner_folder_dir, image_file))
                test_labels.append(folder_label)
                count+=1
            elif count >= sample_size:
                break

        print(folder_name+" Complete")

    #first convert list of list to numpy array
    labels = np.array(labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    #converting to tensors
    # imagepaths1 = imagepaths

    # imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32)
     
    
    # image, label = tf.train.slice_input_producer([imagepaths, labels],shuffle=True)

    # image = tf.read_file(image)
    # image = tf.image.decode_jpeg(image, channels=3) #keeping all 3 RGB channels for now

    # #### customized image processing code will go here
    
    # # kept as defult from reference will be changed later
    # image = tf.image.resize_images(image, [28, 28])
    # image = image * 1.0/127.5 - 1.0

    # # Create batches
    # X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)


    #randomize data
    #imagepaths1 = tf.random_shuffle(imagepaths1, seed=8)
    #labels = tf.random_shuffle(labels, seed=8)

    #create tf.data batchset
    # dataset = tf.data.Dataset.from_tensor_slices((imagepaths1, labels))
    # dataset = dataset.map(_parse_function)
    # dataset = dataset.batch(batch_size)

    # test_dataset = tf.data.Dataset.from_tensor_slices((test_imagepaths, test_labels))
    # test_dataset = test_dataset.map(_parse_function)
    # test_dataset = test_dataset.batch(batch_size)

    return imagepaths, labels, test_imagepaths, test_labels




#####
# @Desc: Loads the image dataset into batches tensors that can be read by tensorflow.
#       
# @params: dataset_dir <str> the dataset directory
#          image_type <str> the file extention of the images to be used
#          batch_size <int> the size of each tensorbatch
#   
# @returns: X --> values of the tensor batches 
#           Y --> labels of the tensor batches
# @References:
#               1. https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
#               2. https://youtu.be/umGJ30-15_A
#               3. http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
#               4. https://www.tensorflow.org/guide/datasets
#####
def prp_dataset_v2(dataset_dir, image_type = '.jpg', batch_size = 100, sample_size=1000, test_perc=0.01):

    imagepaths = [] #Will store the directory of each image 
    labels = []     #Will store a list of one hot encoded labels

    test_imagepaths = []
    test_labels = [] 

    one_hot_dict = one_hot_encoded_label_dict()

    folders = sorted(os.walk(dataset_dir).__next__()[1])

    #traversing the outer Folder
    for folder_name in folders:

        count = 0

        folder_label = one_hot_dict[folder_name.upper()] #The one hot encoded label for the folder's contents 

        inner_folder_dir = os.path.join(dataset_dir, folder_name)
        
        image_files = os.walk(inner_folder_dir).__next__()[2] # [1] for folders, [2] for files, [0] not sure

        #print(len(image_files))

        #traversing the inner directory and looping through every image
        for image_file in image_files:
            
            if image_file.endswith(image_type) and count < sample_size: #ensuring only intended imagetype is stored
                imagepaths.append(os.path.join(inner_folder_dir, image_file))
                labels.append(folder_label)
                count+=1
            elif count >= sample_size:
                break

        print(folder_name+" Complete")


    return imagepaths, labels