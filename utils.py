import os
import io
import pickle
import GLTR
from matplotlib import pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import logging

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
        features = get_VGG19_features('dummy.png')

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
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
    img = image.load_img(img_path, target_size=(224, 224))
    out = image.img_to_array(img)
    out = np.expand_dims(out, axis=0)
    out = preprocess_input(out)
    flatten = model.predict(out)
    return list(flatten[0])