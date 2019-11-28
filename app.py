# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, send_from_directory
import predict
import pandas as pd

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def main_dash():
    message = 'Give text to get prediction'
    pred = ''
    text = ''
    return render_template('index.html', message = message, pred = pred, text = text)


@app.route('/process', methods=['POST'])
def process():
    dataset = request.form['dataset']
    lm = request.form['lm']
    feature_set = request.form['feature_set']
    classifier = request.form['classifier']
    lmot = feature_set.split('_')[0]
    feature_type = '_'.join(feature_set.split('_')[1:])


    input_text = request.form['input_text']

    pred = predict.evaluate_text(input_text, dataset, lm, lmot, feature_type, classifier)
    print('The given text is: ', pred)
    message = 'The given text was: '
    return render_template('index.html', message=message, pred=pred, text = input_text)

@app.route('/tables')
def results_tables():
    all_results = pd.read_csv('all_results.csv', index_col=False)
    return render_template('tables.html', all_results = all_results)


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
