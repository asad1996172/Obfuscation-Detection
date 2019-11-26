import argparse
import sys
import logging
import utils


def evaluate_text(text, dataset_type, language_model, language_model_output_type, feature_type, classifier):
    """
    :param text: Input text for obfuscation detection
    :param dataset_type: The dataset used to train the obfuscation detection model.
    :param language_model: The Language Model used to train the obfuscation detection model.
    :param language_model_output_type: The Language Model output type used to train the obfuscation detection model.
    :param feature_type: The feature type used to train the obfuscation detection model.
    :param classifier: The classifier used to train the obfuscation detection model.
    :return: a tuple containing three things (prediction, probability_obfuscated, probability_evaded)
    """

    features = utils.extract_features(text, language_model, language_model_output_type, feature_type)

    req_model_path = 'models/' + '_'.join(
        [dataset_type, language_model, language_model_output_type, classifier, feature_type])
    clf = utils.load_model(req_model_path)
    prediction = (clf.predict([features]))[0]
    if prediction == 0:
        logging.warning('The input text is ORIGINAL ')
        return "Original"
    else:
        logging.warning('The input text is OBFUSCATED ')
        return "Obfuscated"


def main():
    parser = argparse.ArgumentParser()

    # Input parameters
    parser.add_argument("--text", "-t", help="Input string containing text for obfuscation detection", default='')
    parser.add_argument("--file_path", "-fp", help="Path of file containing text for obfuscation detection", default='')

    # Obfuscation detection parameters
    parser.add_argument("--dataset_type", "-dt", help="Dataset on which the model was trained on", default='')
    parser.add_argument("--language_model", "-lm", help="Language model used to trained the model", default='bert_base')
    parser.add_argument("--language_model_output_type", "-lmot",
                        help="Type of output used from language model for training the model", default='probs')
    parser.add_argument("--feature_type", "-ft", help="Type of feature used for training model", default='vgg19')
    parser.add_argument("--classifier", "-c", help="Classifier used for training model", default='ann')

    # parsing arguments
    args = parser.parse_args()

    text = args.text
    file_path = args.file_path

    dataset_type = args.dataset_type.lower()
    language_model = args.language_model.lower()
    language_model_output_type = args.language_model_output_type.lower()
    feature_type = args.feature_type.lower()
    classifier = args.classifier.lower()

    # error/warning handling
    if (text == '') and (file_path == ''):
        logging.error('Please give either text or file path as input')
        sys.exit()
    if (text != '') and (file_path != ''):
        logging.warning('Both text and file path given as input. Using only text for obfuscation detection')
        file_path = ''
    if dataset_type == '':
        logging.error('Please give a dataset name on which the required model was trained on')
        sys.exit()

    # extracting text required for obfuscation detection
    if text == '':
        text = utils.read_file(file_path)

    evaluate_text(text, dataset_type, language_model, language_model_output_type, feature_type, classifier)


if __name__ == '__main__':
    main()
