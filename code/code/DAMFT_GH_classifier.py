import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import os
from sklearn.neural_network import MLPClassifier


#-------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--generateroot', default='../output/generatedata/', help='path to the generated dataset')
parser.add_argument('--dataname', default='abalone_train.csv', help='file name of dataset')
parser.add_argument('--outf', default='../output/performance/', help='folder to output images and model checkpoints')
opt = parser.parse_args()

def read(filename, path):
    """
    Read data from a CSV file.

    Args:
    - filename (str): The name of the CSV file.
    - path (str): The directory path where the file is located.

    Returns:
    - content (DataFrame): The content of the CSV file.
    - x_data (DataFrame): The input features extracted from the content.
    - y (array): The target labels extracted from the content.

    This function reads the CSV file located at the specified path and extracts the input features (x_data)
    and target labels (y) using the `get_data` function.
    """
    # Construct the full file path
    file = path + filename

    # Read the CSV file into a DataFrame
    content = pd.read_csv(file, header=None)

    # Extract input features (x_data) and target labels (y) using the get_data function
    x_data, y = get_data(content)

    return content, x_data, y


def get_data(content):
    """
    Extracts features (x_data) and labels (y) from the given DataFrame.

    Args:
    - content (DataFrame): DataFrame containing the dataset.

    Returns:
    - x_data (DataFrame): Features of the dataset.
    - y (array): Labels of the dataset.
    """
    x_data = content.iloc[:, :-1]
    y = np.array(content.iloc[:, -1])
    return x_data, y


def predict(classifier, x_test):
    """
    Predict the labels for test data using a classifier.

    Args:
    - classifier: The trained classifier model.
    - x_test (DataFrame): Test data features.

    Returns:
    - y_pre (list): Predicted labels for the test data.

    This function predicts the labels for the test data using the trained classifier model.
    """
    y_pre = []
    for i in range(len(x_test)):
        # Predict the label for each test data instance and append it to the list
        y_pre.append(classifier.predict([x_test.iloc[i]]))
    return y_pre


def performence(y_test, y_pre):
    """
    Calculate performance metrics for classification.

    Args:
    - y_test (array-like): True labels of the test data.
    - y_pre (array-like): Predicted labels for the test data.

    Returns:
    - performance (list): List containing accuracy, mean precision, and mean F1 score.

    This function calculates accuracy, mean precision, and mean F1 score for classification tasks.
    It uses scikit-learn's metrics functions to compute these metrics.

    """
    # Calculate accuracy
    acc = metrics.accuracy_score(y_test, y_pre)
    # Calculate mean precision for each class
    precision = np.mean(metrics.precision_score(y_test, y_pre, average=None))
    # Calculate mean F1 score for each class
    f1 = np.mean(metrics.f1_score(y_test, y_pre, average=None))
    # Create a performance list with calculated metrics
    performance = [acc, precision, f1]
    return performance


def save(perfor):
    """
    Save the performance metrics to a CSV file.

    Args:
    - perfor (list): List containing performance metrics.

    This function saves the performance metrics to a CSV file. It creates the necessary directories if they don't exist.
    The performance metrics are appended to the CSV file with appropriate headers.
    """
    # Define the output directory
    outf = opt.outf
    try:
        os.makedirs(outf)
    except OSError:
        pass

    # Define the path for the performance file
    performance_path = os.path.join(opt.outf + os.path.basename(__file__) + ".csv")

    # Convert the performance metrics to a DataFrame
    perform = pd.DataFrame([perfor])

    # Define the header for the CSV file
    head = ['acc', 'precision', 'f1', 'filename']

    # Check if the performance file already exists
    if not os.path.exists(performance_path):
        # If the file does not exist, save the DataFrame with the header
        perform.to_csv(performance_path, header=head, index=False, mode='a')
    else:
        # If the file exists, append the DataFrame without the header
        perform.to_csv(performance_path, header=False, index=False, mode='a')


def train_classifier(x_train, y_train, x_g, y_g):
    """
    Train a classifier using a two-stage approach with generated and original data.

    Args:
    - x_train (DataFrame): Features of the original training data.
    - y_train (array): Labels of the original training data.
    - x_g (DataFrame): Features of the generated data.
    - y_g (array): Labels of the generated data.

    Returns:
    - mlp (list): Trained classifier model.

    This function trains a classifier using a two-stage approach:
    1. Training with the generated data.
    2. Training with the original training data after initialization with weights and biases obtained from stage 1.
    The trained classifier is returned as a list.
    """
    # Traing with the generated data
    num = int((2 + x_train.shape[1]) * 2 / 3)
    mlp = MLPClassifier(hidden_layer_sizes=(num,), max_iter=30, solver='adam', activation='identity')
    mlp.fit(x_g, y_g)
    w1 = mlp.coefs_[1]
    b1 = mlp.intercepts_[1]
    a = np.matmul(x_train, mlp.coefs_[0]) + mlp.intercepts_[0]

    mlp1 = MLPClassifier(hidden_layer_sizes=(), max_iter=1500, solver='lbfgs')
    mlp1.coefs_ = w1
    mlp1.intercepts_ = b1

    # Training with the original training data
    mlp1.fit(a, y_train)
    #
    print(mlp1.get_params())

    w2 = []
    b2 = []
    w2.append(mlp.coefs_[0])
    w2.append(mlp1.coefs_[0])
    print(mlp.intercepts_)
    print(mlp1.intercepts_)
    b2.append(mlp.intercepts_[0])
    b2.append(mlp1.intercepts_[0])
    mlp.coefs_ = w2
    mlp.intercepts_ = b2

    return mlp


if __name__ == "__main__":
    """
        Main execution script for training classifier with the generated data and original training data.
    """
    numimages_lis = [300, 500, 1000, 2000, 5000]
    for num in numimages_lis:
        # loading data
        dataname = opt.dataname
        fi_test = dataname.replace("train", "test")
        fi_g=dataname[:-4]+str(num)+'.csv'
        # Read training data
        train_x_y, x_train, y_train = read(dataname, opt.dataroot)
        # Read testing data
        test_x_y, x_test, y_test = read(fi_test, opt.dataroot)
        # Read sample data

        g_x_y, x_g, y_g = read(fi_g, opt.generateroot)

        # Train classifier
        classifier = train_classifier(x_train, y_train, x_g, y_g)

        # Predict
        y_pre = predict(classifier, x_test)

        # Evaluate performance
        perfor = performence(y_test, y_pre)
        # Append filename to performance metrics
        perfor.append(dataname)

        # Save performance metrics
        save(perfor)

