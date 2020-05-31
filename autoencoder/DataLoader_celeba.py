import shutil
import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
import pandas as pd

image_path = "../data/celeba/img_align_celeba"
image_ROI_path="../data/celeba/img_align_celeba/ROI"
celeba_attr_file = "../data/celeba/list_attr_celeba.txt"
male_list_file="../data/celeba/male_list.txt"
female_list_file="../data/celeba/female_list.txt"
male_folder="../data/celeba/male"
female_folder="../data/celeba/female"
partition_file="../data/celeba/list_eval_partition.txt"
traindata_file="../data/celeba/celeba-train.csv"
validdata_file="../data/celeba/celeba-valid.csv"
testdata_file="../data/celeba/celeba-test.csv"
ROI_file="../data/celeba/list_ROI.txt"
filter_file="../data/celeba/image_list.txt"

Attr_type = 21 # Male
attr_type = 'Male'
img_height=32
img_width=32
num_classes=2
num_imges_train= 162770
num_imges_valid=19867
num_imges_test=19962
random_seed = 123

#clean and seperate the labels
def label_split():

    count_male=0
    count_female=0
    try:
        os.remove(male_list_file)
        os.remove(female_list_file)
    except:
        print("Error while deleting file %s or %s" % (male_list_file, female_list_file))
    
    with open(celeba_attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
    with open(male_list_file,"a+") as male_list, open(female_list_file, "a+") as female_list:
        for line in Attr_info:
            array = line.split()
            if int(array[Attr_type]) == 1: 
                male_list.write(array[0] + '\n')
                count_male+=1
            elif int(array[Attr_type]) == -1:
                female_list.write(array[0]+'\n')
                count_female+=1
            else: 
                continue

    print ("There are %d males" % (count_male)) 
    print ("There are %d females" % (count_female)) 

def image_split():

    folders=[male_folder, female_folder]
    files=[male_list_file, female_list_file]

    for i in range(len(folders)):
        if not os.path.exists(folders[i]):
            os.mkdir(folders[i])

        with open(files[i],"r") as list_file:
            list_imgs = list_file.readlines()

        for line in list_imgs:
            src_file=os.path.join(image_path,line.rstrip())
            if os.path.isfile(src_file):
                shutil.copyfile(src_file,os.path.join(folders[i],line.rstrip()))
            else:
                print('not found: %s' % src_file)

def img_ROI():
    
    if not os.path.exists(image_ROI_path):
        os.mkdir(image_ROI_path)
    df2 = pd.read_csv(partition_file, sep="\s+", skiprows=0, header=None)
    df2.columns = ['Filename', 'Partition']
    df2 = df2.set_index('Filename')
    for name, row in df2.iterrows():
        if (os.path.splitext(name)[1] != ".jpg"): 
            df2.drop(index=name)
            continue
        image = face_recognition.load_image_file(image_path+'/'+name)
        face_location = face_recognition.face_locations(image)
        if face_location:
            top, right, bottom, left = face_location[0]
            face_image = image[top:bottom, left:right]
            pil_image = PIL.Image.fromarray(face_image)
            pil_image=pil_image.resize((img_width, img_height), PIL.Image.ANTIALIAS)
            pil_image.save('%s/ROI_%s'%(image_ROI_path,name))
        else: 
            df2=df2.drop(index=name)

    df2.to_csv(ROI_file)      
    #plt.imshow(pil_image)
    #plt.show()


#data augmentation to balance data
def data_aug():
    pass

def data_partition():
    #df1 = pd.read_csv(filter_file, sep=",", skiprows=0, header=None)
    #df1.columns=['Filename']
    #df1['Filename']=df1['Filename'].str.replace('ROI_', '', regex=False)
    #print(df1['Filename'].head())

    df2 = pd.read_csv(ROI_file, sep=",", skiprows=0, header=0)
    df2 = df2.set_index('Filename')
    #df2=df2.loc[df1['Filename'],:]
    df2.loc[df2['Partition'] == 0].to_csv(traindata_file)
    df2.loc[df2['Partition'] == 1].to_csv(validdata_file)
    df2.loc[df2['Partition'] == 2].to_csv(testdata_file)

#read imges list into buffer
def data_Loader(male_list, female_list):

    all_list=[male_list, female_list]
    folders=[male_folder, female_folder]
    data_list=[]
    for i in range(len(all_list)):
        for item in all_list[i]:
            img = PIL.Image.open(os.path.join(folders[i], item))
            data = np.asarray(img/255.0, dtype=np.uint8)
            data_list.append(data)
    return data_list

def get_data(name_list):
    data_list=[]
    for item in name_list:
        filename='ROI_'+item
        img = PIL.Image.open(os.path.join(image_ROI_path, filename))
        data = np.asarray(img, dtype=np.uint8)/255.0
        data_list.append(data)
    return np.array(data_list)

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def load_data(index, shuffle=False):
    data_file=[traindata_file, validdata_file, testdata_file]
    num_imges=[num_imges_train, num_imges_valid, num_imges_test]
    with open(celeba_attr_file, "r") as Attr_file:
        num_imgs = Attr_file.readline()
        Attr_header = Attr_file.readline()
    df1 = pd.read_csv(celeba_attr_file, sep="\s+", skiprows=2, header=None)
    df1.columns=['Filename']+Attr_header.split()
    df1 = df1.set_index('Filename')
    #print(df1[attr_type].head())

    df2=pd.read_csv(data_file[index], sep=",", skiprows=0, header=0)
    images=get_data(df2['Filename'])
    labels=df1.loc[df2['Filename'],attr_type]
    if shuffle:
        shuffle_idx = np.arange(num_imges[index])
        shuffle_rng = np.random.RandomState(random_seed)
        shuffle_rng.shuffle(shuffle_idx)
        images, labels=images[shuffle_idx], labels[shuffle_idx]
    return images, labels, one_hot_encoded(class_numbers=labels, num_classes=num_classes)

def train_data(shuffle):
    return load_data(0, shuffle)

def test_data(shuffle):
    return load_data(1, shuffle)

def valid_data(shuffle):
    return load_data(2, shuffle)


if __name__ == "__main__":
    #Loader()
    #label_split()
    #image_split()
    #data_Loader(['000003.jpg'],['000001.jpg'])
    #data_partition()
    #img_ROI()
    #train_data()
    #test_data(True)
    '''img = PIL.Image.open(image_ROI_path+'/ROI_000005.jpg')
    print(np.asarray(img, dtype=np.uint8).shape)
    plt.imshow(img)
    plt.show()'''
    '''df2=pd.read_csv(traindata_file, sep=",", skiprows=0, header=0)
    print(df2.shape[0])
    df2=pd.read_csv(validdata_file, sep=",", skiprows=0, header=0)
    print(df2.shape[0])
    df2=pd.read_csv(testdata_file, sep=",", skiprows=0, header=0)
    print(df2.shape[0])'''
    '''image = face_recognition.load_image_file(image_path+'/'+'000004.jpg')
    face_location = face_recognition.face_locations(image)
    if face_location:
        top, right, bottom, left = face_location[0]
        face_image = image[top:bottom, left:right]
        pil_image = PIL.Image.fromarray(face_image)
        pil_image=pil_image.resize((img_width, img_height), PIL.Image.ANTIALIAS)
        pil_image.save('%s/ROI_%s'%(image_ROI_path,name))'''
    pass