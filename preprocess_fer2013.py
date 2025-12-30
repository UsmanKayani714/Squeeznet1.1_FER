import csv
import os
import numpy as np
import h5py
import skimage.io
import torch

ck_path_train = 'datasets/fer_2013/train'
ck_path_test = 'datasets/fer_2013/test'

anger_path_train = os.path.join(ck_path_train, 'angry')
disgust_path_train = os.path.join(ck_path_train, 'disgust')
fear_path_train = os.path.join(ck_path_train, 'fear')
happy_path_train = os.path.join(ck_path_train, 'happy')
neutral_path_train = os.path.join(ck_path_train, 'neutral')
sadness_path_train = os.path.join(ck_path_train, 'sad')
surprise_path_train = os.path.join(ck_path_train, 'surprise')

anger_path_test = os.path.join(ck_path_test, 'angry')
disgust_path_test = os.path.join(ck_path_test, 'disgust')
fear_path_test = os.path.join(ck_path_test, 'fear')
happy_path_test = os.path.join(ck_path_test, 'happy')
neutral_path_test = os.path.join(ck_path_test, 'neutral')
sadness_path_test = os.path.join(ck_path_test, 'sad')
surprise_path_test = os.path.join(ck_path_test, 'surprise')


# # Creat the list to store the data and label information
data_x_train = []
data_y_train = []
data_x_test = []
data_y_test = []

datapath = os.path.join('KMUtada','fer2013.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(anger_path_train)
files.sort()
print(f"Processing {len(files)} train anger images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(anger_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(0)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train anger images")

files = os.listdir(disgust_path_train)
files.sort()
print(f"Processing {len(files)} train disgust images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(disgust_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(1)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train disgust images")

files = os.listdir(fear_path_train)
files.sort()
print(f"Processing {len(files)} train fear images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(fear_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(2)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train fear images")

files = os.listdir(happy_path_train)
files.sort()
print(f"Processing {len(files)} train happy images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(happy_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(3)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train happy images")

files = os.listdir(neutral_path_train)
files.sort()
print(f"Processing {len(files)} train neutral images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(neutral_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(4)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train neutral images")

files = os.listdir(sadness_path_train)
files.sort()
print(f"Processing {len(files)} train sadness images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(sadness_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(5)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train sadness images")

files = os.listdir(surprise_path_train)
files.sort()
print(f"Processing {len(files)} train surprise images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(surprise_path_train,filename))
    data_x_train.append(I.tolist())
    data_y_train.append(6)
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(files)} train surprise images")

# Test data
files = os.listdir(anger_path_test)
files.sort()
print(f"Processing {len(files)} test anger images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(anger_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(0)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test anger images")

files = os.listdir(disgust_path_test)
files.sort()
print(f"Processing {len(files)} test disgust images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(disgust_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(1)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test disgust images")

files = os.listdir(fear_path_test)
files.sort()
print(f"Processing {len(files)} test fear images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(fear_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(2)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test fear images")

files = os.listdir(happy_path_test)
files.sort()
print(f"Processing {len(files)} test happy images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(happy_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(3)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test happy images")

files = os.listdir(neutral_path_test)
files.sort()
print(f"Processing {len(files)} test neutral images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(neutral_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(4)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test neutral images")

files = os.listdir(sadness_path_test)
files.sort()
print(f"Processing {len(files)} test sadness images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(sadness_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(5)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test sadness images")

files = os.listdir(surprise_path_test)
files.sort()
print(f"Processing {len(files)} test surprise images...")
for i, filename in enumerate(files):
    I = skimage.io.imread(os.path.join(surprise_path_test,filename))
    data_x_test.append(I.tolist())
    data_y_test.append(6)
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(files)} test surprise images")



print(np.shape(data_x_train))
print(np.shape(data_y_train))
print(np.shape(data_x_test))
print(np.shape(data_y_test))

datapath_train = os.path.join('FERTdata','fer2013_train.h5')
datapath_test = os.path.join('FERTdata','fer2013_test.h5')

# Create the FERTdata directory if it doesn't exist
if not os.path.exists('FERTdata'):
    os.makedirs('FERTdata')

datafile_train = h5py.File(datapath_train, 'w')
datafile_train.create_dataset("data_pixel", dtype = 'uint8', data=data_x_train)
datafile_train.create_dataset("data_label", dtype = 'int64', data=data_y_train)
datafile_train.close()

datafile_test = h5py.File(datapath_test, 'w')
datafile_test.create_dataset("data_pixel", dtype = 'uint8', data=data_x_test)
datafile_test.create_dataset("data_label", dtype = 'int64', data=data_y_test)
datafile_test.close()

print("Save data finish!!!")
