# Text Classification Lib on Keras
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

KERAS-TEXT-CLS is a deep learning text classification library based on [KERAS](https://keras.io/). 
It designs an abstract text classification pipeline including tokenizer, 
vocabulary builder, embedding constructor, and deep network classifier.
Users can choose different processors for different text classification tasks. 
In addition, processors in the pipeline can be inherited and customized. 

## Usage

KERAS-TEXT-CLS designs and implements a text classification pipeline including:

### Tokenizer

Tokenizer is to tokenize documents or sentences into tokens or words

There are some implemented or integrated tokenizers,
- Char Tokenizer, which tokenize sentences into characters
- [Jieba Tokenizer](https://github.com/fxsjy/jieba), which is a fast open-source Chinese Tokenizer 

More tokenizers can be added and customized by implementing the abstract methods from BaseTokenizer.  

### Vocabulary Builder

Vocabulary builder is to build the vocabulary based on tokenized corpus, and convert tokens into bag-of-word indices.

### Embedding Constructor
 
Embedding Constructor is to construct the pre-trained embedding matrix for deep network models.

Currently, tow embedding methods are integrated,
- [Word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [FastText](https://pypi.org/project/fasttext/)

More embedding algorithm can be added and customized by implementing the abstract methods from BaseEmbedder.

### Model

Model is to train and predict the processed text. It inherits from the keras subclassing model. 

Currently four state-of-art text classification models are implemented based on 
[keras model subclassing](https://keras.io/models/about-keras-models/#model-subclassing).

- MLP - Multi-Layler Perceptron

- TextCNN - Convolutional Neutral Network [1]

- RCNN - Recurrent Convolutional Neural Network [2]

- HAN - Hierarchical attention networks [3]

Some more latest algorithm and models would be implemented and added in the future.


## Examples
```python
from keras_text_cls.model import *
from keras_text_cls.tokenizer import *
from keras_text_cls.vocab import *
from keras_text_cls.embedding import *

text = ['我爱北京天安门，天安门上太阳升',
        '上海是全国的科技、贸易、信息、金融和航运中心']
labels = np.array([[0, 1], [1, 0]])

# Init a tokenizer
jieba_tokenizer = JiebaTokenizer()
# Tokenize documents into tokens
words = jieba_tokenizer.text_to_words(text)

# Init the vocabulary
vocabulary = Vocabulary()
# Build bag-of-words on the given words
vocabulary.fit_on_words(words)
# Convert words into bag-of-word indices
indices = vocabulary.words_to_idx(words)

# Padding the text indices to the length of max_sequence
max_seq = 10
inputs = pad_text_indices(indices, max_seq)

# Init the keras model TextMLP
text_mlp = TextMLP(num_classes=2,
                   embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                   dropout=0.1,
                   num_hidden_units=[10], max_seq_len=max_seq, multi_label=False)

# Compile the keras model
text_mlp.compile(loss='categorical_crossentropy',
        optimizer='adam',
      metrics=['acc'])

# Train the keras model 
text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
# Prediction 
y_prob = text_mlp.predict(inputs)
y_pred = (y_prob >= 0.5).astype(int)

```


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