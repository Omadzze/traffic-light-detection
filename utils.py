import numpy as np
import yaml
import os
import random
import shutil
import cv2
import matplotlib.pyplot as plt
import filecmp
import time
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from ultralytics import YOLO







##### Training Core Functions #####

 



# Uncertainty

def calculate_uncertainty_naive(confidences, 
                          threshold):
    
    '''


    Input:  
    - confidences: List of traffic light confidence scores
    - threshold: the value from which we will consider the confidences to calculate uncertainty

    Output:
    '''
    
    under_n = sum([1 for elem in confidences if elem < threshold])
    print(f'Confidences under {threshold}: {under_n}/{len(confidences)}')
    uncertainty_score = under_n / len(confidences)

    return uncertainty_score



def calculate_uncertainty(traffic_light_confidences,
                          other_class_confidences, 
                          threshold, 
                          multiplier):
    
    '''


    Input:  
    - traffic_light_confidences: List of traffic light confidence scores
    - other_class_confidences: List of all other classes confidence scores
    - threshold: the value from which we will consider the confidences to calculate uncertainty
    - multiplier: the weight multiplier to traffic light label

    Output:
    '''
    
    traffic_lights_n = sum([1 for elem in traffic_light_confidences if elem < threshold])
    other_classes_n = sum([1 for elem in other_class_confidences if elem < threshold])

    weighted_traffic_lights = traffic_lights_n * multiplier
    total_count = weighted_traffic_lights + other_classes_n
    
    uncertainty_score = total_count / (len(traffic_light_confidences) * multiplier + len(other_class_confidences))
    
    
    return uncertainty_score



def get_batch(results, 
              image_paths, 
              threshold = 0.1,
              n_top = 10):
    
    '''
    Get the n_top batch of most uncertain elements of the results object (given by yolo)

    Input: 
    - results: result object from YOLO predict function
    - image_path: the path where the pool of unlabeled images is located
    - threshold:  (see calculate_uncertainty)
    - multiplier: (see calculate_uncertainty)
    - n_top: the number of top most uncertain elements we want to retrieve

    Output  
    - The n_top most uncertain element's IDs from the results object
    '''

    print(f'Get batch with {len(results)} results')

    uncertainty_scores = []

    # Iteration through results (from model assessment):
    for result in results:
        confidences = []
        boxes = result.boxes  # Boxes object for bbox outputs
        objects = boxes.data[:, 4:6] # Confidence and class ID of the detected objects

        # Iteration through objects:
        for object in objects:
            conf = object.data[0]
            # class_id = object.data[1]
            confidences.append(conf)


        if len(confidences) != 0:
            uncertainty = calculate_uncertainty_naive(confidences, threshold)
        else:
            uncertainty = 0
        uncertainty_scores.append(uncertainty)

    # n_top = 2
    # indices = sorted(range(len(uncertainty_scores)), key=lambda i: uncertainty_scores[i], reverse=True)[:n_top]
    indices = sorted(range(len(uncertainty_scores)), key = lambda sub: uncertainty_scores[sub], reverse = True)[:n_top]


    # Print out top n uncertain elements (function form Basic Utility Functions [see below])
    print_top_n_uncertain(image_paths, uncertainty_scores, indices, n_top)
    
    selected = [image_paths[i].split('/')[-1][:-4] for i in indices]
    
    return selected



def uncertainty_sampling(model, 
                         device, 
                         data_path, 
                         val_data_path, 
                         test_data_path, 
                         batch_size, 
                         conf = 0.01):
    
    '''
    This function runs uncertainty measure to select the top sample_size most uncertain samples. 
    It then creates temporary directories in the necessary form and hierarchy that is required to train a YOLOv8 model
    as well as the necessary (temporary) configuration yaml file.
    
    Input: 
    - model: model for probabilities
    - data_path: path to data
    - batch_size: sample size
    - train_split: train split ratio
    
    Output: 
    - path to temporary training configuration yaml file
    '''

    try:
        os.mkdir('tmp')
        
        # Images
        os.mkdir('tmp/images')
        os.mkdir('tmp/images/train')

        # Labels
        os.mkdir('tmp/labels')
        os.mkdir('tmp/labels/train')

        
    except FileExistsError:
        print(f'Directory already exists!')

    t1 = time.time()   
    image_paths = [os.path.join(data_path, 'images', filename) for filename in os.listdir(os.path.join(data_path, 'images')) if os.path.isfile(os.path.join(data_path, 'images',filename))]
    t2 = time.time()
    print(f'Image path list created. (Time: {t2 -t1}s)')

    t1 = time.time()
    results = model(image_paths, 
                    conf = conf, 
                    save = False,
                    verbose = False,
                    device = device) ### MODIFY?  e.g. add confidence ###
    t2 = time.time()
    print(f'\nModel results on images generated. (Time: {t2 -t1}s)\n')
    
    t1 = time.time()
    train_set = get_batch(results, image_paths, n_top = batch_size)
    t2 = time.time()
    print(f'\nBatch of images selected. (Time: {t2 -t1}s)\n')

    for elem in train_set:
        shutil.copy(f'{data_path}/labels/{elem}.txt', f"tmp/labels/train/{elem}.txt")
        shutil.copy(f'{data_path}/images/{elem}.png', f"tmp/images/train/{elem}.png")
        
    path = os.path.join(os.getcwd(), 'tmp')

    config = {
        'path': path,
        'train': os.path.join(path, 'images/train'),
        'val': os.path.join(val_data_path, 'images'),
        'test': os.path.join(test_data_path, 'images'),
        'names': {0: 'traffic light'}
    }
    
    yaml_path = os.path.join(os.getcwd(), 'tmp.yaml')
    
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style = False)
     
    
    return yaml_path, train_set





# Diversity

# Function to perform k-means clustering on image features
def perform_kmeans_clustering(features,
                              k):
    """
    Perform k-menas clustering on image labels (bounding boxes)

    Parameters:
    - features (numpy array): Labels files extracted a numpy array
    - k (int): Number of clusters

    Returns:
    - cluster_labels (numpy array): Cluster that were assigned for the each bounding boxes
    """

    kmeans = KMeans(n_clusters = k, random_state = 42)
    cluster_labels = kmeans.fit_predict(features)
    print("Cluster labels after clustering:")
    print(cluster_labels)
    return cluster_labels



# Function to select representative samples from each cluster
def select_representative_samples(cluster_labels,
                                  k,
                                  image_paths):

    """
    The functions is providing representative sample from each cluster. So ensures inclusion of images that we
    got from k-means clustering

    Input:
    cluster_labels (numpy array): Cluster that were assigned for the each bounding boxes
    k (int): The number of clusters
    image_paths (list): A list of paths to each image file

    Output:
    cluster_samples (dictionary): keys - is a cluster indices and values - randomly selected images from each cluster
    """
    cluster_samples = {}

    for cluster_id in range(k):
        cluster_indices = (cluster_labels == cluster_id).nonzero()[0]

        if len(cluster_indices) > 0:
            selected_sample = random.choice(cluster_indices)

            # Ensure that selected_sample is within the valid range
            selected_sample = min(selected_sample, len(image_paths) - 1)

            # Get the image name corresponding to the selected sample
            image_name = os.path.basename(image_paths[selected_sample])

            # Store the cluster index along with the image name
            cluster_samples[cluster_id] = image_name
            print("Cluster Image")
            print(cluster_samples[cluster_id])

    return cluster_samples



def extract_features_from_images(image_paths,
                                 label_folder):


    """
    This function is extracting values that we have in label files. So it extracting bounding boxes and
    then creates numpy array of this values

    Input:
    image_paths (list): A list of paths to each image file
    label_folder (string): Path to the labels folder

    Output:
    features (list): Returns the list of numpy array with the bounding boxes values
    """


    features = []
    for image_path in image_paths:
        # Assuming label files have the same name as image files but with a different extension
        label_path = os.path.join(label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        # Read label file and extract bounding box information
        bounding_boxes = read_label_file(label_path)

        # Assuming bounding_boxes is a list of lists where each inner list contains [x_center, y_center, width, height]
        for box in bounding_boxes:
            # Append the box coordinates as features
            features.append(box)

    return np.array(features)



def read_label_file(label_path):

    """
    This function is reading every label file in particullar coordinates and then append them to the
    list called bounding boxes

    Input:
    label_path (string): Path to the each label file

    Output:
    bounding_boxes (list): List with the bounidng boxes values [x_center, y_center, width, height]
    """

    bounding_boxes = []

    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                _, x_center, y_center, width, height = map(float, line.split())
                bounding_boxes.append([x_center, y_center, width, height])
    except FileNotFoundError:
        print(f"File not found: {label_path}")

    return bounding_boxes



# Function to perform diversity sampling using k-means clustering
def diversity_sampling(data_path,
                       image_paths,
                       k,
                       val_data_path,
                       test_data_path):

    """
    Main function that performs diversity sampling, creates yaml file

    Input:
    data_path(string): path to the datapool
    image_paths (list): path to the each image
    k (int): The number of
    val_data_path (string): path to the validation datapool
    test_data_path (string): path to the test datapool

    Output:
    yaml_path (string): Path to the yaml file
    train_set (list): list of image ids in train set
    """

    unique_image_ids = set()

    label_folder = os.path.join(data_path, "labels")

    features = extract_features_from_images(image_paths, label_folder)
    # Perform k-means clustering
    cluster_labels = perform_kmeans_clustering(features, k)

    # Select representative samples from each cluster
    representative_samples = select_representative_samples(cluster_labels, k, image_paths)

    representative_samples_copy = representative_samples.copy()

    # Create temporary directories and copy selected samples
    try:
        os.mkdir('diversity')

        os.mkdir('diversity/images')
        os.mkdir('diversity/images/train')

        os.mkdir('diversity/labels')
        os.mkdir('diversity/labels/train')

    except FileExistsError:
        print(f'Directory already exists!')

    print(representative_samples_copy)

    # Extract the image IDs from the dictionary values
    for ID in list(representative_samples_copy.values()):
        image_id = ID[:-4]
        unique_image_ids.add(image_id)
    #train_set = [ID[:-4] for ID in list(representative_samples_copy.values())]
    #print(train_set)

    print("LIST:")
    print(unique_image_ids)

    # Copy images and labels for the train set
    for elem in unique_image_ids:
        shutil.copy(f'{data_path}/labels/{elem}.txt', f"diversity/labels/train/{elem}.txt")
        shutil.copy(f'{data_path}/images/{elem}.png', f"diversity/images/train/{elem}.png")

    path = os.path.join(os.getcwd(), 'diversity')

    config = {
        'path': path,
        'train': os.path.join(path, 'images/train'),
        'val': os.path.join(val_data_path, 'images'),
        'test': os.path.join(test_data_path, 'images'),
        'names': {0: 'traffic light'}
    }

    yaml_path = os.path.join(os.getcwd(), 'diversity.yaml')

    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style = False)
    except:
        print("Error: File does not appear to exist.")

    return yaml_path, unique_image_ids





# Random

def random_sampling(data_path, 
                    val_data_path, 
                    test_data_path, 
                    batch_size):
    
    '''
    This function selects random training samples to some sample size and train/test-split. 
    It then creates temporary directories in the necessary form and hierarchy that is required to train a YOLOv8 model as well as the necessary (temporary) configuration yaml file.
    
    Input: 
    - data_path: path to data
    - batch_size: sample size
    
    Output: 
    - path to temporary training configuration yaml file
    '''
    
    files = os.listdir(os.path.join(data_path, "labels"))
    selected = random.sample(files, batch_size)
    train_set = [item[:-4] for item in selected]
    
    print(train_set)
    
    try:
        os.mkdir('tmp')
        
        # Images
        os.mkdir('tmp/images')
        os.mkdir('tmp/images/train')

        # Labels
        os.mkdir('tmp/labels')
        os.mkdir('tmp/labels/train')

        
    except FileExistsError:
        print(f'Directory already exists!')

    for elem in train_set:
        shutil.copy(f'{data_path}/labels/{elem}.txt', f"tmp/labels/train/{elem}.txt")
        shutil.copy(f'{data_path}/images/{elem}.png', f"tmp/images/train/{elem}.png")
    
    path = os.path.join(os.getcwd(), 'tmp')
    
    config = {
        'path': path,
        'train': os.path.join(path, 'images/train'),
        'val': os.path.join(val_data_path, 'images'),
        'test': os.path.join(test_data_path, 'images'),
        'names': {0: 'traffic light'}
    }
    
    yaml_path = os.path.join(os.getcwd(), 'tmp.yaml')
    
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style = False)
     
    
    return yaml_path, train_set





# Training Loop

def training_loop(model,
                  device, 
                  al_budget, 
                  datapool_path, 
                  train_data_path, 
                  val_data_path, 
                  test_data_path, 
                  batch_size = 5, 
                  epochs = 5, 
                  batch_train = True, 
                  mode = 0):
    
    '''
    Trains a given model on randomly chosen samples using the other functions from utils.py
    
    Input: 
    - al_budget: number of allowed images to train on
    - model: model
    - datapool_path: path to datapool folder
    - train_data_path: path to training data folder
    - test_data_path: path to fixed test data folder
    - train_split: ratio of training data
    - batch_size: sample size
    - epochs: number of epochs to train for

    Output: 
    - None
    '''

    
    
    #### Random Selection
    if(mode == 0):
        
        ### Batch Training
        
        if(batch_train):
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                print(f'Remaining budget: {al_budget}')

                # Select batch of images and create temporary yaml files
                tmp_yaml, image_IDs = random_sampling(data_path = datapool_path,
                                                      test_data_path = test_data_path,
                                                      val_data_path = val_data_path,
                                                      batch_size = batch_size)
                
                # Train only on the selected batch of images
                res = model.train(data = tmp_yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                
                model = YOLO(f'{res.save_dir}/weights/best.pt')
                
                # Delete all temporary directories & files
                delete_tmp('tmp')
                
                # Move the batch of images from the datapool to the training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)

        ### Training on the whole dataset
        
        else:
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                
                # Create the yaml file (if it doesn't exist already)
                yaml = create_yolo_config_file(train_data_path,
                                               val_data_path,
                                               test_data_path, 
                                               ['traffic light'])
                
                # Select batch of images
                _, image_IDs = random_sampling(data_path = datapool_path,
                                                      test_data_path = test_data_path,
                                                      val_data_path = val_data_path,
                                                      batch_size = batch_size)
                
                # Delete all temporary directories & files
                delete_tmp('tmp')
                
                # Move selected batch of images from datapool to training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)
                
                # Train model on the whole training dataset with the added batch
                res = model.train(data = yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                
                model = YOLO(f'{res.save_dir}/weights/best.pt')
            

    #### Active Learning Selection (Uncertainty):    
    elif(mode == 1):
        
        ### Batch Training
        
        if(batch_train):
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                
                # Select batch of images and create temporary yaml files
                tmp_yaml, image_IDs = uncertainty_sampling(model,
                                                         device = device,
                                                         data_path = datapool_path,
                                                         val_data_path = val_data_path,
                                                         test_data_path = test_data_path,
                                                         batch_size = batch_size)
                
                # Train only on the selected batch of images
                res = model.train(data = tmp_yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                
                model = YOLO(f'{res.save_dir}/weights/best.pt')

                
                # Delete all temporary directories & files
                delete_tmp('tmp')
                
                # Move the batch of images from the datapool to the training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)
            
                
        ### Training on the whole dataset
        
        else:
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                
                # Create the yaml file (if it doesn't exist already)
                yaml = create_yolo_config_file(train_data_path,
                                               val_data_path,
                                               test_data_path, 
                                               ['traffic light'])
                
                # Select batch of images
                _, image_IDs = uncertainty_sampling(model,
                                                  device = device,
                                                  val_data_path = val_data_path,
                                                  data_path = datapool_path,
                                                  test_data_path = test_data_path,
                                                  batch_size = batch_size)
                
                # Delete all temporary directories & files
                delete_tmp('tmp')
                
                # Move selected batch of images from datapool to training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)
                
                # Train model on the whole training dataset with the added batch
                res = model.train(data = yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                
                model = YOLO(f'{res.save_dir}/weights/best.pt')
            

    elif(mode == 2):

        ### Batch Training
        if(batch_train):
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                
                # Select batch of images and create temporary yaml files
                image_paths = [os.path.join(datapool_path, "images", filename) for filename in
                           os.listdir(os.path.join(datapool_path, "images"))]

                # Call the diversity_sampling function
                tmp_yaml, image_IDs = diversity_sampling(data_path = datapool_path,
                                                         image_paths = image_paths,
                                                         k = 5,
                                                         val_data_path = val_data_path, 
                                                         test_data_path = test_data_path)

                # Train only on the selected batch of images
                res = model.train(data = tmp_yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                model = YOLO(f'{res.save_dir}/weights/best.pt')
                

                # Delete all temporary directories & files
                delete_tmp('diversity')
                
                # Move the batch of images from the datapool to the training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)
            
                
        ### Training on the whole dataset
        
        else:
            
            ## Training until our budget is exhausted
            while al_budget > 0:
                
                # Substract used samples from budget / set sample size to remaining budget if there remains less budget than our sample size
                if al_budget < batch_size:
                    batch_size = al_budget
                al_budget -= batch_size
                
                # Create the yaml file (if it doesn't exist already)
                yaml = create_yolo_config_file(train_data_path, 
                                               val_data_path, 
                                               test_data_path, 
                                               ['traffic light'])
                
                # Select batch of images
                _, image_IDs = diversity_sampling(data_path = datapool_path,
                                                  image_paths = image_paths,
                                                  k = 12,
                                                  val_data_path = val_data_path, 
                                                  test_data_path = test_data_path)
                
                # Delete all temporary directories & files
                delete_tmp('diversity')
                
                # Move selected batch of images from datapool to training dataset
                move_to_trainset(datapool_path, 
                                 train_data_path, 
                                 image_IDs)
                
                # Train model on the whole training dataset with the added batch
                res = model.train(data = yaml, 
                            epochs = epochs, 
                            device = device, 
                            single_cls = True)
                model = YOLO(f'{res.save_dir}/weights/best.pt')
                

    else:
        print("mode not supported yet")
        return        
        
    return model





##### Basic Utility Functions #####



def convert_to_yolo_format(yaml_path, 
                           image_folder, 
                           output_folder):
    
    '''
    Converts a yaml file containing image labels (in the format of BOSCH small traffic light dataset) into the format required by YOLO (v8).
    
    Input: 
    - yaml_path: path to YOLO training configuration yaml file
    - image_folder: path to image folder
    - output_folder: path to output folder
    
    Output: 
    - txt file for each image ID containing all the respective bounding boxes in the YOLO format (x_center, y_center, width, height)
    '''
    
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    for entry in data:
        image_path = entry['path']
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image_file_path = os.path.join(image_folder, os.path.basename(image_path))

        output_file_path = os.path.join(output_folder, f"{image_id}.txt")

        # Check if the output file already exists
        if os.path.exists(output_file_path):
            continue

        # Get image dimensions
        image_width, image_height = get_image_dimensions(image_file_path)

        with open(output_file_path, 'w') as out_file:
            for box in entry['boxes']:
                
                x_min, y_min, x_max, y_max = float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])
            
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width/2
                y_center = y_min + height/2
            
                # Normalizing and border handling in x-direction
                if x_center >= 0.0 and x_center <= image_width:
                    x_center /= image_width
                    width /= image_width
                elif x_center < 0.0:
                    x_center = 0.0
                    width = (width - (abs(x_center) + width/2)) / image_width
                else:
                    x_center = 1.0 # image_width / image_width
                    width = (width - (abs(x_center) + width/2)) / image_width
                
                # Normalizing and border handling in y-direction:
                if y_center >= 0.0 and y_center <= image_height:
                    y_center /= image_height
                    height /= image_height
                elif y_center < 0.0:
                    y_center = 0.0
                    height = (height - (abs(y_center) + height/2)) / image_height
                else:
                    y_center = 1.0 # image_height / image_height
                    height = (height - (abs(y_center) + height/2)) / image_height
                

                # Write to the YOLO format file
                out_file.write(f"0 {x_center} {y_center} {width} {height}\n")



def get_image_dimensions(image_path):
    
    '''
    Returns the dimensions of a given image.
    
    Input: 
    - image_path: path to image
    
    Output: 
    - width
    - height of the image
    '''
    
    with Image.open(image_path) as img:
        
        
        return img.size


                      
def check_negative_labels(folder_path):
    
    '''
    Checks a directory containing all the label txt files for the images for negative values in the labels.
    
    Input: 
    - folder_path: path to label directory
    
    Output: 
    - list of files containing negative values
    '''
    
    files_with_negative_values = [] # Initialize list with files containing negative values
    
    # Putting all the files into a list
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    # Iterate through the files
    for file_name in files:
        
        file_path = os.path.join(folder_path, file_name) # Path to the txt files
        
        with open(file_path, "r") as file: # Open the txt file
            
            lines = file.readlines() # Create list of all the lines in txt file
            
            for line in lines: # Iteration through the lines
                
                values = line.split() # Splitting the lines at the space () between the values (creates list of the different values)
                
                for value in values[1:]: # Checking for any negative value in values
                    
                    if float(value) < 0:
                        
                        print(value)
                        
                        files_with_negative_values.append(file_name) # Append filename containing negative value
                        
                        break # No need to look further as one negative value is sufficient
            
    
    return files_with_negative_values



def plot_bounding_boxes(image_id, 
                        images_folder, 
                        labels_folder):
    
    '''
    Displays an image with plotted bounding boxes.
    
    Input: 
    - image_id: image ID
    - images_folder: path to image folder
    - labels_folder: path to label folder
    
    Output: 
    - None
    '''
    
    # Load image
    image_path = os.path.join(images_folder, f"{image_id}.png")
    
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        

    # Load labels
        labels_path = os.path.join(labels_folder, f"{image_id}.txt")

        with open(labels_path, 'r') as file:
            
            lines = file.readlines()

            for line in lines:
                
                class_idx, x_center, y_center, width, height = map(float, line.split())
                
                print(f"X: {x_center}, Y: {y_center}, Width: {width}, Height: {height}")
                
                # Get image dimensions
                image_width, image_height = get_image_dimensions(image_path)

                # Calculate bounding box coordinates & denormalize
                x1 = (x_center - width / 2) * image_width
                y1 = (y_center - height / 2) * image_height
                x2 = (x_center + width / 2)  * image_width
                y2 = (y_center + height / 2) * image_height

                draw.rectangle((x1, y1, x2, y2), outline="red")

        # Display the image with bounding boxes
        img.show()
        
        
        
def delete_tmp(name):
    
    '''
    Deletes all temporary files and directories.
    
    Input: 
    - None
    
    Output: 
    - None
    '''

    shutil.rmtree(name)
    os.remove(f'{name}.yaml')
      
    
        
def print_top_n_uncertain(image_paths, 
                          uncertainty_scores, 
                          indices, 
                          n):
    
    '''
    Prints top n uncertain elements.
    
    Input: 
    - path to images
    - uncertainty scores
    - indices, number of top elements
    
    Output: 
    - None
    '''
    
    
    for i in range(n):
        if i < len(indices):
            index = indices[i]
            print(f"    {i+1}. id {image_paths[index].split('/')[-1][:-4]} --> {uncertainty_scores[index]}")
        else:
            break
            
            

def create_env(folder_path, 
               image_folder, 
               output_folder):
    
    '''
    Input: 
    path folder with yaml file, also with rgb/trains
    
    Output: txt file for each image ID containing all the respective bounding boxes in the YOLO format (x_center, y_center, width, height)
    ''' 

    yaml_path = folder_path + '/train.yaml'

    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    for entry in data:
        image_path = entry['path']
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image_file_path = os.path.join(image_folder, os.path.basename(image_path))
        shutil.copyfile(folder_path + '/' + image_path, image_file_path)
        output_file_path = os.path.join(output_folder, f"{image_id}.txt")

        # Check if the output file already exists
        if os.path.exists(output_file_path):
            continue

        # Get image dimensions
        image_width, image_height = get_image_dimensions(image_file_path)

        with open(output_file_path, 'w') as out_file:
            for box in entry['boxes']:
                
                x_min, y_min, x_max, y_max = float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])
            
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width/2
                y_center = y_min + height/2
            
                # Normalizing and border handling in x-direction
                if x_center >= 0.0 and x_center <= image_width:
                    x_center /= image_width
                    width /= image_width
                elif x_center < 0.0:
                    x_center = 0.0
                    width = (width - (abs(x_center) + width/2)) / image_width
                else:
                    x_center = 1.0 # image_width / image_width
                    width = (width - (abs(x_center) + width/2)) / image_width
                
                # Normalizing and border handling in y-direction:
                if y_center >= 0.0 and y_center <= image_height:
                    y_center /= image_height
                    height /= image_height
                elif y_center < 0.0:
                    y_center = 0.0
                    height = (height - (abs(y_center) + height/2)) / image_height
                else:
                    y_center = 1.0 # image_height / image_height
                    height = (height - (abs(y_center) + height/2)) / image_height
                

                # Write to the YOLO format file
                out_file.write(f"0 {x_center} {y_center} {width} {height}\n")
                
                
                
def copy_directory(original_directory, 
                   new_directory):
    '''
    Recursively copy a directory and its contents to a new directory. (this could be used especially for testing the following functions without messing up any dataset)
    
    Input:
    - original_directory: Path to the original directory to be copied
    - new_directory: Path to the new directory where the contents will be copied
    
    Output: None
    '''

    # Check if the original directory exists
    if not os.path.exists(original_directory):
        print(f"Original directory '{original_directory}' does not exist.")
        return

    # Check if the new directory already exists
    if os.path.exists(new_directory):
        print(f"New directory '{new_directory}' already exists.")
        return

    # Create the new directory
    os.makedirs(new_directory)

    # Iterate over the items (files and subdirectories) in the original directory
    for item in os.listdir(original_directory):
        item_path = os.path.join(original_directory, item)
        new_item_path = os.path.join(new_directory, item)

        if os.path.isdir(item_path):
            # If it's a subdirectory, recursively copy it
            copy_directory(item_path, new_item_path)
        else:
            # If it's a file, copy it to the new directory
            shutil.copy2(item_path, new_item_path)

    print(f"Directory '{original_directory}' and its contents copied to '{new_directory}'.")

    
    
def create_new_set(images_path, 
                    labels_path, 
                    new_set, 
                    ratio = 0.2):
    '''
    Create a new set by randomly selecting images and labels from the original dataset and moving them to a new directory.
    
    Input:
    - images_path: Path to the original images folder.
    - labels_path: Path to the original labels folder.
    - new_set: Name of the directory where the new set will be created.
    - ratio: Ratio of the original dataset used for the new set.
    
    Output: 
    - absolute path to the new set
    '''

    # Create the test set directory if it doesn't exist
    if not os.path.exists(new_set):
        os.makedirs(new_set)

    # Create new directories for the test set images and labels
    new_images_folder = os.path.join(new_set, 'images')
    new_labels_folder = os.path.join(new_set, 'labels')

    if not os.path.exists(new_images_folder):
        os.makedirs(new_images_folder)
    if not os.path.exists(new_labels_folder):
        os.makedirs(new_labels_folder)

    # List all image files in the original images folder
    image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    num_images_to_select = int(len(image_files) * ratio)

    selected_image_files = random.sample(image_files, num_images_to_select)

    for image_file in selected_image_files:
        image_path = os.path.join(images_path, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)

        # Movew the image and label to the test set directory
        shutil.move(image_path, new_images_folder)
        shutil.move(label_path, new_labels_folder)

    print(f"Created a new set with {num_images_to_select} images in '{new_set}' directory.")
    
    workdir = os.getcwd()
    
    return os.path.join(workdir, new_set)



def create_yolo_config_file(dataset_path, 
                            val_data_path, 
                            test_data_path, 
                            classes):
    
    '''
    Creates a YOLO training configuration yaml file, given some dataset and the classes we want to train on.
    
    Input: 
    - dataset_path: absolute path to dataset folder
    - val_data_path: path to validation data folder
    - test_data_path: path to test data folder
    - classes: list of classes
   
    Output:
    - path to yaml file
    '''
    
    
    
    config_data = {
        "path": dataset_path,
        "train": os.path.join(dataset_path, 'images'),
        "val": os.path.join(val_data_path, 'images'),
        "test": os.path.join(test_data_path, 'images'),
        "names": {i: class_name for i, class_name in enumerate(classes)}
    }

    work_dir = os.getcwd()
    config_file_path = os.path.join(work_dir, 'config.yaml')
    
    # Skip creating file if it already exists
    if os.path.exists(config_file_path):
        print(f"Config file available: {config_file_path}")
    
    # Create file if it doesn't exist
    else:
        with open(config_file_path, 'w') as config_file:
            yaml.dump(config_data, config_file, default_flow_style = False)
            print(f"Created YOLO config file: {config_file_path}")
    
    return config_file_path

                
       
def move_to_trainset(datapool_path, 
                     train_data_path, 
                     image_IDs):
    
    '''
    Moves a batch of selected samples from Datapool to Training Dataset.
    
    Input: 
    - datapool_path: path to datapool directory
    - train_data_path: path to training data directory
    - image_IDs: list of image IDs to be moved (e.g. defined by "selected" which can be returned from the select functions above)
    
    Output: 
    - None
    '''
    
    datapool_images = os.path.join(datapool_path, 'images')
    datapool_labels = os.path.join(datapool_path, 'labels')
    
    training_images = os.path.join(train_data_path, 'images')
    training_labels = os.path.join(train_data_path, 'labels')
    
    for ID in image_IDs:
        img = ID + '.png'
        label = ID + '.txt'
        img_path = os.path.join(datapool_images, img)
        print(img_path)
        label_path = os.path.join(datapool_labels, label)
        print(label_path)
        
        shutil.move(img_path, training_images)
        shutil.move(label_path, training_labels)
        
        
        
        
def display_img(img_path, 
                figsize = (10, 10)):
    
    '''
    Displays an image.
    
    Input:
    - img_path: path to image
    - figsize: size of the plot for display
    
    Output:
    - None
    '''
    
    img = np.array(cv2.imread(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # colour channels from image are different from those used by matplotlib
    fig, ax = plt.subplots(figsize = figsize)
    ax.imshow(img)
    ax.axis('off')
    
    
    
def compare_directories(dir1, 
                        dir2):
    
    '''
    Checks for file duplicates between two directories.
    
    Input:
    - dir1: first directory
    - dir2: second directory
    
    Output:
    - list of duplicate files that exist in both directories
    '''
    
    # Lists to store file names
    files_in_dir1 = set(os.listdir(dir1))
    files_in_dir2 = set(os.listdir(dir2))

    # Find common files in both directories
    common_files = files_in_dir1.intersection(files_in_dir2)
    identical_files = []

    # Compare the content of each common file
    for file in common_files:
        file1 = os.path.join(dir1, file)
        file2 = os.path.join(dir2, file)

        # If the files are identical, add to the list
        if filecmp.cmp(file1, file2, shallow = False):
            identical_files.append(file)

    return identical_files
