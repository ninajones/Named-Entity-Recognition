# Import statements
import pickle
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
from itertools import groupby
from typing import List, Tuple, Dict
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


"""
Load in tags, set batch size, set max sentence length, create NN with elmo embedding layer.
Load in trained model.
"""


# Global variables

tags = None
n_tags = None
model = None


# Import tags

def _import_tags():
    global tags
    global n_tags

    with open('tags.pickle', 'rb') as f:
        tags = pickle.load(f)

    n_tags = len(tags)


# Define model

def _define_model():
    global model

    sess = tf.Session()
    K.set_session(sess)

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    def _elmo_embedding(x):
        return elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(batch_size * [max_len])
        },
            signature="tokens",
            as_dict=True)["elmo"]

    max_len = 200
    batch_size = 32

    input_text = Input(shape=(max_len,), dtype=tf.string)
    embedding = Lambda(_elmo_embedding, output_shape=(max_len, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(embedding)
    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    model = Model(input_text, out)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.load_weights('./testmodel_weights')


# Setup function

def setup():
    _import_tags()
    _define_model()


"""
Tokenize and add word and sentence padding to prediction dataset.
Predict tags and make them readable.
"""


# Add word padding to article

def _add_word_padding(words: List[str], words_per_sent = 200, pad_str ='__PAD__') -> List[str]:
    num_pad_words = words_per_sent - len(words)
    pad_words = ['__PAD__'] * num_pad_words
    return words + pad_words


# Add sentence padding to article

def _gen_pad_sentences(sentences: List[List[str]], words_per_sent = 200, sent_per_article = 512, pad_str ='__PAD__') -> List[List[str]]:
    pad_sent = ['__PAD__'] * words_per_sent
    num_pad_sentences = sent_per_article - len(sentences)
    pad_sentences = [pad_sent] * num_pad_sentences
    return pad_sentences


# Tokenize article sentences into words and then apply word and sentence padding functions

def _create_nn_input(article: str) -> List:
    # Split article into list of sentences
    sentences: List[str] = sent_tokenize(article)

    # Split sentences into list of words
    sentences_split_into_words = [word_tokenize(sentence) for sentence in sentences]

    # Add padding to words
    pad_words = [_add_word_padding(sentence) for sentence in sentences_split_into_words]

    # Add padding to sentences
    return pad_words + _gen_pad_sentences(pad_words)


# Group tags representing same entity together

def _group_tags(x):
    # Change list of tuples into list
    nested_list = [item for t in x for item in t]

    # Add string 'split' before each tag containing B in 0th index of tag
    res = []
    for entry in nested_list:
        if entry[0:2] == 'B-':
            res.append('split')
        res.append(entry)
    nested_list[:] = res

    # Split by string 'split'
    grouper = groupby(res, key=lambda x: x in {'split'})

    # Convert to dictionary via enumerate
    conv_to_dict = dict(enumerate((list(j) for i, j in grouper if not i), 1))

    # Convert dictionary of lists into list of lists
    dictionary_to_list = [[k] + v for k, v in conv_to_dict.items()]

    # Remove first element in each list
    for x in dictionary_to_list:
        del x[0]

    return dictionary_to_list


# Change tag type for B if group contains two I's, then drop the I's. Otherwise, just drop the I's.

def _change_B(x):
    result = x

    I_tags = list(filter(lambda x: x[:2] == 'I-', x))
    if len(I_tags) == 2:
        second_to_last_I_tag = I_tags[-2]
        new_B_tag = 'B' + second_to_last_I_tag[1:]
        bb_no_Is = list(filter(lambda x: x[:2] != 'I-', x))
        result = bb_no_Is.copy()
        result[0] = new_B_tag
    else:
        bb_no_Is = list(filter(lambda x: x[:2] != 'I-', x))
        result = bb_no_Is.copy()

    return result


# Change B- tags with more readable label names

def _change_label(label):
    if label == 'B-geo':
        return 'Location'
    elif label == 'B-gpe':
        return 'Geopolitical Entity'
    elif label == 'B-org':
        return 'Company'
    elif label == 'B-per':
        return 'Person'
    elif label == 'B-tim':
        return 'Time Period'
    else: return label


# Apply _change_label function to first element of each list

def _give_human_readable_label(grouping: List[str]) -> List[str]:
    label = [_change_label(grouping[0])]
    elements = grouping[1:]

    return label + elements


# Join all but first element of each list into string

def _join_elements(x):
    return [x[0]] + [' '.join(x[1:])]


# Merge lists with same first element and create dictionary

def _merge_lists(x):

    d = defaultdict(list)

    for i, j in x:
        d[i].append(j)

    return d


# Apply tag formatting functions to each entity

def _make_tags_readable(tags: List[Tuple]) -> Dict[str, List[str]]:
    grouped_tags = _group_tags(tags)

    # remove I-s for each element of grouped_tags
    grouping_i_removed = [_change_B(grouping) for grouping in grouped_tags]

    readable_groupings = [_give_human_readable_label(grouping) for grouping in grouping_i_removed]

    joined_groupings = [_join_elements(grouping) for grouping in readable_groupings]

    return _merge_lists(joined_groupings)


# Predict tags and apply tag formatting

def predict_tags(article):
    # create list of lists
    preprocessed_article = np.array(_create_nn_input(article))
    predictions = model.predict(preprocessed_article)
    # picks best tag for each word
    predicted_label_indices = np.argmax(predictions, axis=-1)

    # gets tags of each element of a
    convert_tags = np.vectorize(lambda x: tags[x])
    # apply convert_tags on predicted_label_indices
    predicted_tags = convert_tags(predicted_label_indices)

    # 2d array for labels create pairwise tuples with 2d array of words
    flat_words = preprocessed_article.flatten()
    flat_tags = predicted_tags.flatten()

    combine_tags_words = list(zip(flat_tags, flat_words))

    # function, for each tuple (lbl, word) in result, keep tuples where label (t[0]) not O
    raw_tags = list(filter(lambda t: t[0] != 'O', combine_tags_words))

    return _make_tags_readable(raw_tags)


# For testing

if __name__ == "__main__":
    print("Setting up")
    setup()

    print("Setup complete")

    article = """Google is a company. Bob Jones is a person."""

    print("Extracting Entities")
    print("Entities: ")
    print(predict_tags(article))
