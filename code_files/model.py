import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scikitplot as skplt
import sys
from PIL import Image

#function to set cuda usage
def set_device():
    if torch.cuda.is_available:
       dev = 'cuda:3'            # which cuda device to be used for training the model on GPU
    else:
       dev = 'cpu'
    return torch.device(dev)

# function to get mean and standard deviation of the tensors from EEG Images
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)   #in current batch
        # print(images.shape)
        images = images.view(image_count_in_a_batch, images.size(1), -1) # need to reshape to get the mean and std
        # print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count
    return mean, std


# function to show transformed images if one wants to have a look of teh images before and after normalization
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels: ', labels)


#function to evaluate the model on test set
def evaluate_model_on_test_set(model, test_loader):
    lst_all_predicted, lst_all_labels = [], []
    # switch the model from training mode to evaluation mode
    model.eval()  # will notify all your layers that now we are in validation mode, in this way the dropout will work in the evaluation mode instead of training mode
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    # deactivate the auto-gradient engine to reduce th ememory usage and speed up the computations however we will not be able to back propagate
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(
                0)  # to keep track of how many images we have in total as the last batch may have less than declared batch size

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce

            predicted_correctly_on_epoch += (predicted == labels).sum().item()
            for i in range(len(predicted.tolist())):
                lst_all_predicted.append(predicted.tolist()[i])  # tolist() converts tensor to list
                lst_all_labels.append(labels.tolist()[i])

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("     - Testing dataset.  Got %d out of %d images correctly (%.3f%%). "
          % (predicted_correctly_on_epoch, total, epoch_acc))
    return lst_all_labels, lst_all_predicted

#function to find average of last three epoch results(P, R, F1, Macro AVg P, Macro AVg R, Macro AVg F1)
def avg_scores_last_3_epochs(lst_dct):
    avg_P_s, avg_R_s, avg_F1_s, avg_P_ns, avg_R_ns, avg_F1_ns, avg_macro_P, avg_macro_R, avg_macro_F1 = 0,0,0,0,0,0,0,0,0
    for i in range(3):
        avg_P_ns += lst_dct[-(i+1)]['0']['precision']
        avg_R_ns += lst_dct[-(i+1)]['0']['recall']
        avg_F1_ns += lst_dct[-(i+1)]['0']['f1-score']
        avg_P_s += lst_dct[-(i+1)]['1']['precision']
        avg_R_s += lst_dct[-(i+1)]['1']['recall']
        avg_F1_s += lst_dct[-(i+1)]['1']['f1-score']
        avg_macro_P += lst_dct[-(i+1)]['macro avg']['precision']
        avg_macro_R += lst_dct[-(i+1)]['macro avg']['recall']
        avg_macro_F1 += lst_dct[-(i+1)]['macro avg']['f1-score']
    avg_P_ns=round(avg_P_ns/3,2)
    avg_R_ns=round(avg_R_ns/3,2)
    avg_F1_ns=round(avg_F1_ns/3,2)
    avg_P_s=round(avg_P_s/3,2)
    avg_R_s=round(avg_R_s/3,2)
    avg_F1_s=round(avg_F1_s/3,2)
    avg_macro_P=round(avg_macro_P/3,2)
    avg_macro_R=round(avg_macro_R/3,2)
    avg_macro_F1=round(avg_macro_F1/3,2)
    return avg_P_s,avg_R_s,avg_F1_s,avg_P_ns,avg_R_ns,avg_F1_ns,avg_macro_P,avg_macro_R,avg_macro_F1

# train the model while returning result dict for averaging the results of last 3 epochs
def train_nn_avg_last_3_epochs(model, train_loader, test_loader, criterion, optimizer, n_epochs):  # criterion = loss_fn
    device = set_device()
    lst_dct = []
    global y_true, y_pred
    for epoch in range(n_epochs):
        print('Epoch Number %d' % (epoch + 1))
        model.train()
        # to see how many images were classified correctly
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # now iterate through all of the batches
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # before starting backpropagation we set grad to 0 so parameters gets updated correctly
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        # check how many % of images were classified correctly
        epoch_acc = 100.00 * running_correct / total

        print("     - Training dataset.  Got %d out of %d images correctly (%.3f%%). Epoch loss:  %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        y_true, y_pred = evaluate_model_on_test_set(model, test_loader)
        dct = classification_report(y_true, y_pred, output_dict=True)
        lst_dct.append(dct)

    print('Finished! ')
    return lst_dct

# model training
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):  # criterion = loss_fn
    device = set_device()
    global y_true, y_pred
    for epoch in range(n_epochs):
        print('Epoch Number %d' % (epoch + 1))
        model.train()
        # to see how many images were classified correctly
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # now iterate through all of the batches
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # before starting backpropagation we set grad to 0 so parameters gets updated correctly
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        # check how many % of images were classified correctly
        epoch_acc = 100.00 * running_correct / total

        print("     - Training dataset.  Got %d out of %d images correctly (%.3f%%). Epoch loss:  %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        y_true, y_pred = evaluate_model_on_test_set(model, test_loader)

    print('Finished! ')
    return model

#function to set model train test loader and other model parameters
def set_model(train_dataset_path, test_dataset_path, mod_name, inp_size, transform_flag, num_classes, b_size):
    model_name = mod_name

    # Training Transformations with optional augmentations
    training_transforms = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),  # Ensure consistent input size
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    # Test transforms without augmentation
    test_transforms = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=False)

    # Calculate Dataset Mean and Std
    mean, std = get_mean_and_std(train_loader)
    normalized_transforms = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Update datasets with normalization
    train_dataset.transform = normalized_transforms
    test_dataset.transform = normalized_transforms

    # Load Pre-trained Vision Transformer
    model = create_model(model_name, pretrained=True, num_classes=num_classes)

    # Add Dropout to the model head (if available)
    if hasattr(model, 'head'):
        model.head = nn.Sequential(
            model.head,
            nn.Dropout(p=0.3)  # Dropout rate for regularization Resize:
        )

    device = set_device()
    print('Using Device:', device)
    model = model.to(device)

    # Define Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.005)

    return model, train_loader, test_loader, loss_fn, optimizer

# function to oversample the seizure onset training set(minority class)
def oversample_minority_class(train_dataset_path):
    # Oversample the minority(seizure onset) class of the train set to avoid any bias
    ### to be run only once for oversampling of the minority class
    sz_train_dataset_path = train_dataset_path + '/sz'  # assuming "sz" is the folder which contains all seizure onset training images
    ##copying the files 15 times to equalize both classes as the seizure-onsets:non-seizure is in ratio 1:15.
    for fin in glob.glob(sz_train_dataset_path + '/*'):
        for i in range(15):
            new_fin = fin.replace('.png', '__' + str(i + 1) + '.png')
            shutil.copy(fin, new_fin)



#function to run the model and see average of last 3 epoch results
def run_model_with_avg_epochs_results(train_dataset_path, test_dataset_path, num_epochs):
    oversample_minority_class(train_dataset_path)
    #setting the model for run
    num_classes = 2
    ViT_model, train_loader, test_loader, loss_fn, optimizer = set_model('/media/data/fol/visor/data_files/seiz_prop/visor/pat_spf_espf_10/gen_10_pat/train',
                                                                     '/media/data/fol/visor/data_files/seiz_prop/visor/pat_spf_espf_10/gen_10_pat/test',
                                                                     'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',448,True,2,4)
    lst_dct = train_nn_avg_last_3_epochs(ViT_model, train_loader, test_loader, loss_fn, optimizer, num_epochs)


# function to run model and print last epoch results
def run_model(train_dataset_path, test_dataset_path, num_epochs):
    num_classes = 2
    ViT_model, train_loader, test_loader, loss_fn, optimizer = set_model(train_dataset_path, test_dataset_path,
                                                                     'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',448,True,2,4)
    y_true, y_pred, target_names = [], [], ['class_' + str(i + 1) for i in range(2)]
    train_nn(ViT_model, train_loader, test_loader, loss_fn, optimizer, num_epochs)
    print('\nClassification Report for non-seiz(label -> 0) vs seiz(label -> 1): \n\n',classification_report(y_true, y_pred))


if __name__ == '__main__':
    # Check if correct number of arguments is provided
    if len(sys.argv) < 3:
        print('Please at least give train_dataset_path and test_dataset_path to proceed!')
        print("Usage: python3 model.py train_dataset_path test_dataset_path num_epochs")
        sys.exit()
    else:
        try:
            train_dataset_path = sys.argv[1]
            test_dataset_path = sys.argv[2]
            num_epochs = sys.argv[3]      # by default 10
        except Exception as ex:
            if len(sys.argv) == 3:
                num_epochs = 10  # by default we keep the number of epochs to be 10
            else:
                print('Please at least give train_dataset_path and test_dataset_path to proceed!')
                print("Usage: python3 model.py train_dataset_path test_dataset_path num_epochs")
                sys.exit()
    # run the vit model and print average of last 3 epoch results
    run_model_with_avg_epochs_results(train_dataset_path, test_dataset_path, num_epochs)
    # run the vit model and print last epoch results
    run_model(train_dataset_path, test_dataset_path, num_epochs)
