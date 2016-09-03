# Visual Speech Recognition (AdeNet)
This page provides instructions to install the necessary packages to run the 
experiments described in the project on Visual Speech Recognition using Deep Learning.

## Installing
To run the codes, the following dependencies are required: 

1. miniconda2
2. matplotlib
3. pydotplus
4. tabulate
5. scikit-learn 6. ipython
7. pillow
8. theano
9. lasagne
10. nolearn

It is suggested that you use miniconda to setup a virtual environment before running the codes 
to prevent the packages from messing up with your current python environment. 
Miniconda can be download from http://conda.pydata.org/miniconda.html. 
To install the necessary dependencies you can use the following bash script:

```
#!/bin/bash
./Miniconda2−latest−Linux−x86 64.sh
conda create −n ip−avsr python source activate ip−avsr
 
pip install pip install pip install pip install pip install pip install
matplotlib pydotplus tabulate scikit −learn ipython pillow
pip install −−upgrade https://github.com/Theano/Theano/archive/master.zip 
pip install −−upgrade https://github.com/Lasagne/Lasagne/archive/master.
zip
pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn
==0.7.git
```

which creates a virtual environment ip-avsr, activates the virtual environment and installs all 
the necessary python packages to this virtual environment.

## Code Structure
The source codes for different datasets are separated into individual folders named based on 
dataset (`avletters, ouluvs, cuave`). All learning models can be found in the folder `modelzoo` and 
can be imported to code as a python package. Custom neural network layers can be found in the 
package `custom_layers` and the `utils` package contains utility functions such as plotting, 
drawing network layers and image preprocessing functions for normalization and computing delta coefficients.

## Datasets
Within each dataset folder, the codes are further grouped into 3 folders. The data folder contains 
all the mouth ROIs, DCT features and Image Differences extracted for the individual dataset. 
The format used is MatLab’s `.mat` format to allow interchangeability between MatLab and python as the 
pretraining stage requires the use of MatLab DBN code.
The model folder contains all pretrained, finetuned and trained networks so they can be easily reloaded 
in future without the need to retrain them from scratch. The config folder contains a list of `.ini` config files 
that are used for different models (**DeltaNet, AdeNet v1, AdeNet v2**). A list of options are provided below. 
The training programs are called unimodal.py, bimodal.py, trimodal.py for single stream, double stream 
and triple stream input source respectively. 
All training codes accepts a config file using the option `--config`. Type `python trimodal.py -h` to see usage options.

```
usage: trimodal.py [−h] [−−config CONFIG] [−−write results WRITERESULTS]
optional arguments:
−h, −−help show this help message and exit
−−config CONFIG config file to use, default=config/trimodal.ini
−−write results WRITE RESULTS write results to file
```

## Config File Options
Under the `data` section:
- images: raw image ROIs used to extract DBNFs.
- dct: dct features with delta coefficients appended.
- diff: diff image ROIs used for difference of image input source.

Under the `models` section:
- pretrained: pretrained DBNF extractor DBN network for raw images.
- finetuned: finetuned DBNF extractor DBN network for raw images.
- pretrained diff: finetuned DBNF extractor DBN network for difference of images.
- finetuned diff: finetuned DBNF extractor DBN network for difference of images.
- fusiontype: the fusion method to use to combine different input sources. 

Under the `training` section:
- learning rate: learning rate to use train the model.
- decay rate: learning rate decay at each epoch after decay start.
- decay start: epoch to start learning rate decay
- do finetune: to perform finetuning of DBNF extractor.
- save finetune: save finetuned model of raw image DBNF extractor.
- load finetune: load finetuned model of raw image DBNF extractor.
- load finetune diff: load finetuned model of image differences DBNF ex- tractor.
- output units: number of output classes.
- lstm units: number of hidden units used in the LSTM classifiers.