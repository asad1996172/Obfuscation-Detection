# ObfsPrints: Obfuscation Detection Tool
ObfsPrints takes text or text file's path as input and using its 640 unique pre-trained models, tells if the input text was written by humans or automated authorship obfuscation tools.

## Getting Started

The following instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

To run this project you should have a disk space of 17GB and python 3 installed.

### Installing

Following are the steps you need to follow to successfully run this.

First clone this repo

```
git clone https://github.com/asad1996172/Obfuscation-Detection
```

Then install all required libraries

```
pip3 install -r requirements.txt
```

Then download pre-trained models and GPT-2 345M model and put them in the working directory. These folders go by the name 'models' and 'output' 

To download models, use the following link
```
https://www.dropbox.com/s/nikke8387y9smtp/models.zip?dl=0
```

To download output, use the following link
```
https://www.dropbox.com/s/5gputd5v1apfjkb/output.zip?dl=0
```

## Running the tool

In order to run the tool, simply run the following command

```
python3 app.py
```

## Built With

* [Flask](http://flask.palletsprojects.com/en/1.1.x/) - The web framework used
* [Scikit Learn](https://scikit-learn.org/stable/) - Machine Learning library used

## License

This project is licensed under the MIT License.

## Acknowledgments

* [Sb Admin 2](https://startbootstrap.com/themes/sb-admin-2/) - Used this template for Front-end
* [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) - Code for GPT-2 and BERT used as Language Models

