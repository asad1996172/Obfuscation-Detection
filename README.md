# Automated Authorship Obfuscation Detection
Authorship attribution aims to identify the author  of  a  text  based  on  stylometric  analysis. Authorship  obfuscation,  on  the  other  hand, aims to protect against authorship attribution by modifying a text’s style.  In this paper, we evaluate the stealthiness of state-of-the-art authorship obfuscation approaches using neural language models in an adversarial setting.  An obfuscator is stealthy to the extent that an adversary finds it challenging to detect whether its output document is original or not - a decision that is key to an adversary.  We show that the leading authorship obfuscation approaches are not stealthy as the output documents can be identified with average F1 score of 0.871. The reason for this weakness is that the obfuscators degrade text smoothness in a predictable i.e., detectable manner.  Our results highlight the need to develop stealthy authorship obfuscation approaches that better protect an author seeking anonymity.

## Demo
![Tool Demo](https://github.com/asad1996172/Obfuscation-Detection/blob/master/final_demo.gif)

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


## Acknowledgments

* [Sb Admin 2](https://startbootstrap.com/themes/sb-admin-2/) - Used this template for Front-end
* [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) - Code for GPT-2 and BERT used as Language Models

