# Text Classification Lib on Keras
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

KERAS-TEXT-CLS is a deep learning text classification library based on keras. 
It designs a abstract text classification pipeline including tokenizer, 
vocabulary builder, embedding constructor, and deep network classifier.
Users can choose different processors for different text classification tasks. 
In addition, processors in the pipeline can be inherited and customized. 

## Usage

## Examples


## Text Classification Models

Currently four state-of-art text classification models are implemented based on 
[keras model subclassing](https://keras.io/models/about-keras-models/#model-subclassing).

- MLP - Multi-Layler Perceptron

- TextCNN - Convolutional Neutral Network [1]

- RCNN - Recurrent Convolutional Neural Network [2]

- HAN - Hierarchical attention networks [3]

Some more latest algorithm and models would be implemented and added in the future.

## Benchmark

Benchmark on public datasets is to be added.


## Build and install from source

`python setup.py sdist`

`pip install dist/keras_text_cls-<version>.tar.gz`

## Requirements

python >= 3.6


## License

KERAS-TEXT-CLS is available under MIT License


## References

[1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).

[2] Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." Twenty-ninth AAAI conference on artificial intelligence. 2015.

[3] Yang, Zichao, et al. "Hierarchical attention networks for document classification." 
Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.