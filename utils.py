import os
import io
import pickle
import GLTR
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
os.environ['KERAS_BACKEND'] = 'theano'
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import logging
from shutil import copyfile
import sys
import pandas as pd

def read_all_results_file():
    all_res_path = '/home/amahmood1/Dropbox/Obfuscation_Detection/Obfuscation_Detection/scripts/old_scripts/all_results.csv'
    all_data = pd.read_csv(all_res_path, index_col=False)
    all_data.Feature = all_data.Feature.replace('Bins0.005size_output', 'Bins 0.005')
    all_data.Feature = all_data.Feature.replace('Bins0.010size_output', 'Bins 0.010')
    all_data.Feature = all_data.Feature.replace('Bins0.001size_output', 'Bins 0.001')
    all_data.Feature = all_data.Feature.replace('sorted_VGG19_features_output', 'VGG19')
    all_data.Feature = all_data.Feature.replace('Bins10size_output', 'Bins 10')
    all_data.Feature = all_data.Feature.replace('Bins50size_output', 'Bins 50')
    all_data.Feature = all_data.Feature.replace('Bins100size_output', 'Bins 100')

    all_data.drop(columns=['Feature_type', 'MCC'], inplace=True)

    all_data.to_csv('all_results.csv', index=None)

def get_list_of_files(dir_name):
    """
    Takes in the directory name and outputs all the files in sub directories
    """

    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files


def read_model_files():
    """
    Transfers pre-trained models in the tools directory
    """

    models_old_path = sys.argv[1]

    all_model_files = []
    all_files = get_list_of_files(models_old_path)
    for file in all_files:
        if '.pkl' in file:
            all_model_files.append(file)

    for model_file in all_model_files:
        orig_model_file = model_file
        model_file = model_file.split('/')[5:]

        if model_file[0] == 'amt_obfuscated':
            model_file[0] = 'ebg_obfuscated'
        elif model_file[0] == 'amt_evaded':
            model_file[0] = 'ebg_evaded'

        if model_file[3] == 'nn':
            model_file[3] = 'ann'
        elif model_file[3] == 'bayesian':
            model_file[3] = 'gnb'

        if model_file[4] == 'sorted_VGG19_features.pkl':
            model_file[4] = 'vgg19.pkl'

        else:
            if model_file[2] == 'ranks':
                if model_file[4] == 'Bins10size.pkl':
                    model_file[4] = 'bins_10.pkl'
                elif model_file[4] == 'Bins50size.pkl':
                    model_file[4] = 'bins_50.pkl'
                elif model_file[4] == 'Bins100size.pkl':
                    model_file[4] = 'bins_100.pkl'
            elif model_file[2] == 'probs':
                if model_file[4] == 'Bins10size.pkl':
                    model_file[4] = 'bins_0.001.pkl'
                elif model_file[4] == 'Bins50size.pkl':
                    model_file[4] = 'bins_0.005.pkl'
                elif model_file[4] == 'Bins100size.pkl':
                    model_file[4] = 'bins_0.010.pkl'

        new_model_file_name = ('_'.join(model_file)).lower()

        print(orig_model_file)
        print(new_model_file_name)
        copyfile(orig_model_file, 'models/' + new_model_file_name)

# read_model_files()
def get_ranks_and_probs(payloads_list):
    """
    :param payloads_list: list containing output from the language model
    :return: ranks and probabilities extracted by language model
    """

    ranks = []
    probs = []
    for payload in payloads_list:
        for (rank, prob) in payload['real_topk']:
            ranks.append(rank)
            probs.append(prob)
    return ranks, probs


def get_list_of_files(dir_name):
    """
    :param dir_name:  Name of the directory for which you need the list of all files
    :return: all_files: List of all files in the given directory
    """

    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)
    return all_files


def read_file(file_path):
    """
    :param file_path:  file path from which text is required
    :return: text: text in the input file path
    """

    text = io.open(file_path, "r", errors="ignore").readlines()
    text = ''.join(str(e) + "" for e in text)
    text = text.strip()

    return text


def load_model(model_path):
    """
    :param model_path: path of model to load
    :return: loaded model
    """

    with open(model_path + '.pkl', 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def extract_features(text, language_model, output_type, feature_type):
    """
    :param text: text for which the language model is required to extract outputs from
    :param language_model: type of language model to be used
    :param output_type: type of language model output to be returned
    :param feature_type: type of feature to be returned
    :return: values for features as a list
    """

    if 'bert' in language_model:
        max_vocab_size = 28996
    elif 'gpt2' in language_model:
        max_vocab_size = 50257

    logging.warning('Extracting output from language models !! Don\'t stop execution ...')
    payloads = GLTR.main_code(text, language_model)
    ranks, probs = get_ranks_and_probs(payloads)
    logging.warning('Extracting features !! Don\'t stop execution ...')
    if output_type == 'probs':
        probs_create_plots(sorted(probs, reverse=True), 'dummy.png')
    elif output_type == 'ranks':
        ranks_create_plots(sorted(ranks, reverse=True), max_vocab_size, 'dummy.png')

    if 'vgg' in feature_type:
        features = get_vgg19_features('dummy.png')
    else:
        if output_type == 'probs':
            bin_size = float((feature_type.split('_'))[-1])
            if bin_size not in [0.001, 0.005, 0.010]:
                logging.error('Wrong Bin size given for probabilities. Give either 0.001, 0.005 or 0.010')
                sys.exit()
            features = get_binned_features_for_probs(probs, bin_size)
        elif output_type == 'ranks':
            bin_size = int((feature_type.split('_'))[-1])
            if bin_size not in [10, 50, 100]:
                logging.error('Wrong Bin size given for ranks. Give either 10, 50 or 100')
                sys.exit()
            features = get_binned_features_for_ranks(ranks, max_vocab_size, bin_size)


    os.remove("dummy.png")
    return features


def ranks_create_plots(ranks, maxsize, name):
    """
    :param ranks: list of ranks from text
    :param maxsize: maximum rank
    :param name: image name to be saved
    """
    x = list(range(1, len(ranks) + 1))
    y = ranks
    plt.axis('off')
    plt.plot(x, y, color='#05D865')
    plt.ylim(0, maxsize)
    plt.savefig(name)
    plt.close('all')


def probs_create_plots(probs, name):
    """
    :param probs: list of probs from text
    :param name: image name to be saved
    """
    x = list(range(1, len(probs) + 1))
    y = probs
    plt.axis('off')
    plt.plot(x, y, color='#05D865')
    plt.ylim(0, 1)
    plt.savefig(name)
    plt.close('all')

def get_vgg19_features(img_path):
    """
    :param img_path: path of image for which we want VGG-19 based features
    :return: features from flatten layer
    """
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
    img = image.load_img(img_path, target_size=(224, 224))
    out = image.img_to_array(img)
    out = np.expand_dims(out, axis=0)
    out = preprocess_input(out)
    flatten = model.predict(out)
    return list(flatten[0])

def get_binned_features_for_ranks(features, max_size, bin_size=50):
    """
    :param features: ranks of input text
    :param max_size: max rank achievable
    :param bin_size: size of bins
    :return: binned features
    """
    feature_list = []
    for val in range(bin_size, max_size, bin_size):
        range_min = val - bin_size
        range_max = val
        count = 0
        for feature in features:
            if (feature >= range_min) and (feature < range_max):
                count+=1
        feature_list.append(count/len(features))
    return feature_list

def get_binned_features_for_probs(features, bin_size=0.001):
    """
    :param features: probabilities of input text
    :param bin_size: size of bin
    :return: binned features
    """
    feature_list = []
    val = bin_size
    while val <= 1:
        range_min = val - bin_size
        range_max = val
        count = 0
        for feature in features:
            if (feature >= range_min) and (feature < range_max):
                count+=1
        feature_list.append(count/len(features))
        val += bin_size

    return feature_list