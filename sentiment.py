## This is the script to run the experiment, this is the script we did deploy on a google cloud server and it
## outputs into results.csv and is executed using a .sh script.
# the original code is retrieved from https://github.com/intel-analytics/analytics-zoo/blob/master/apps/sentiment-analysis/sentiment.ipynb

import pickle
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', help="Model type")
parser.add_argument('--epochs', help="Model type")
parser.add_argument('--iteration_number', help="Model type")
parser.add_argument('--n_test', help="Model type")
parser.add_argument('--n_train', help="Model type")
parser.add_argument('--max_words', help="Model type")
parser.add_argument('--min_words', help="Model type")

args = parser.parse_args()
n_test = int(args.n_test)
n_train = int(args.n_train)
from IPython import get_ipython
from bigdl.dataset import base
import numpy as np


def download_imdb(dest_dir):
    file_name = "imdb.npz"
    file_abs_path = base.maybe_download(file_name,
                                        dest_dir,
                                        'https://s3.amazonaws.com/text-datasets/imdb.npz')
    return file_abs_path


def load_imdb(dest_dir='/tmp/.bigdl/dataset'):
    path = download_imdb(dest_dir)
    f = np.load(path, allow_pickle=True)
    p = np.random.permutation(len(f['x_train']))
    q = np.random.permutation(len(f['x_test']))
    x_train = f['x_train'][p][:n_train]
    y_train = f['y_train'][p][:n_train]
    x_test = f['x_test'][q][:n_test]
    y_test = f['y_test'][q][:n_test]
    f.close()

    return (x_train, y_train), (x_test, y_test)


print('Processing text dataset')
(x_train, y_train), (x_test, y_test) = load_imdb()
print('finished processing text')

import json


def get_word_index(dest_dir='/tmp/.bigdl/dataset', ):
    file_name = "imdb_word_index.json"
    path = base.maybe_download(file_name,
                               dest_dir,
                               source_url='https://s3.amazonaws.com/text-datasets/imdb_word_index.json')
    f = open(path)
    data = json.load(f)
    f.close()
    return data


if (os.path.isfile("util/idx_word.txt")):
    file = open('util/idx_word.txt', 'r')
    idx_word = pickle.load(file)
else:
    print('Processing vocabulary')
    word_idx = get_word_index()
    idx_word = {v: k for k, v in word_idx.items()}
    print('finished processing vocabulary')
    file = open('util/idx_word.txt', 'w')
    pickle.dump(idx_word, file)
    file.close()

def replace_oov(x, oov_char, max_words, min_words):
    return [oov_char if w >= max_words or w < min_words else w for w in x]


def pad_sequence(x, fill_value, length):
    if len(x) >= length:
        return x[(len(x) - length):]
    else:
        return [fill_value] * (length - len(x)) + x


def to_sample(features, label):
    return Sample.from_ndarray(np.array(features, dtype='float'), np.array(label))


padding_value = 1
start_char = 2
oov_char = 3
index_from = 3
max_words = int(args.max_words)
min_words = int(args.min_words)
sequence_len = 500

print('start transformation')

from zoo.common.nncontext import *

sc = init_nncontext("Sentiment Analysis Example")

train_rdd = sc.parallelize(zip(x_train, y_train), 2).map(
    lambda record: ([start_char] + [w + index_from for w in record[0]], record[1])).map(
    lambda record: (replace_oov(record[0], oov_char, max_words, min_words), record[1])).map(
    lambda record: (pad_sequence(record[0], padding_value, sequence_len), record[1])).map(
    lambda record: to_sample(record[0], record[1]))
test_rdd = sc.parallelize(zip(x_test, y_test), 2).map(
    lambda record: ([start_char] + [w + index_from for w in record[0]], record[1])).map(
    lambda record: (replace_oov(record[0], oov_char, max_words, min_words), record[1])).map(
    lambda record: (pad_sequence(record[0], padding_value, sequence_len), record[1])).map(
    lambda record: to_sample(record[0], record[1]))

print('finish transformation')


from bigdl.dataset import news20
import itertools

embedding_dim = 100

if (os.path.isfile("util/glove.txt")):
    file = open('util/glove.txt', 'r')
    glove = pickle.load(file)
else:
    print('loading glove')
    glove = news20.get_glove_w2v(source_dir='/tmp/.bigdl/dataset', dim=embedding_dim)
    print('finish loading glove')
    file = open('util/glove.txt', 'w')
    pickle.dump(glove, file)
    file.close()

print('processing glove')
w2v = [glove.get(idx_word.get(i - index_from), np.random.uniform(-0.05, 0.05, embedding_dim))
       for i in range(1, max_words + 1)]
w2v = np.array(list(itertools.chain(*np.array(w2v, dtype='float'))), dtype='float').reshape([max_words, embedding_dim])
print('finish processing glove')

from bigdl.nn.layer import *

p = 0.2

def build_model(w2v):
    model = Sequential()

    embedding = LookupTable(max_words, embedding_dim)
    embedding.set_weights([w2v])
    model.add(embedding)
    if model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128, p))) \
            .add(Select(2, -1))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128, p))) \
            .add(Select(2, -1))
    elif model_type.lower() == "bi_lstm":
        model.add(BiRecurrent(CAddTable())
                  .add(LSTM(embedding_dim, 128, p))) \
            .add(Select(2, -1))
    elif model_type.lower() == "cnn":
        model.add(Transpose([(2, 3)])).add(Dropout(p)).add(Reshape([embedding_dim, 1, sequence_len])).add(
            SpatialConvolution(embedding_dim, 128, 5, 1)).add(ReLU()).add(
            SpatialMaxPooling(sequence_len - 5 + 1, 1, 1, 1)).add(Reshape([128]))
    elif model_type.lower() == "cnn_lstm":
        model.add(Transpose([(2, 3)])).add(Dropout(p)).add(Reshape([embedding_dim, 1, sequence_len])).add(
            SpatialConvolution(embedding_dim, 64, 5, 1)).add(ReLU()).add(SpatialMaxPooling(4, 1, 1, 1)).add(
            Squeeze(3)).add(Transpose([(2, 3)])).add(Recurrent()
                                                     .add(LSTM(64, 128, p))) \
            .add(Select(2, -1))

    model.add(Linear(128, 100)).add(Dropout(0.2)).add(ReLU()).add(Linear(100, 1)).add(Sigmoid())

    return model

from bigdl.optim.optimizer import *
from bigdl.nn.criterion import *

# max_epoch = 4
max_epoch = int(args.epochs)
batch_size = 16
model_type = args.model_type

optimizer = Optimizer(
    model=build_model(w2v),
    training_rdd=train_rdd,
    criterion=BCECriterion(),
    end_trigger=MaxEpoch(max_epoch),
    batch_size=batch_size,
    optim_method=Adam())

optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=test_rdd,
    trigger=EveryEpoch(),
    val_method=Top1Accuracy())

import datetime as dt

logdir = '/tmp/.bigdl/'
app_name = 'adam-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

train_summary = TrainSummary(log_dir=logdir, app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir=logdir, app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)

import time

start_time = time.time()
train_model = optimizer.optimize()
total_time = (time.time() - start_time)

predictions = train_model.predict(test_rdd)

def map_predict_label(l):
    if l > 0.5:
        return 1
    else:
        return 0

def map_groundtruth_label(l):
    return l.to_ndarray()[0]

y_pred = np.array([map_predict_label(s) for s in predictions.collect()])
y_true = np.array([map_groundtruth_label(s.label) for s in test_rdd.collect()])

correct = 0
for i in range(0, y_pred.size):
    if (y_pred[i] == y_true[i]):
        correct += 1

accuracy = float(correct) / y_pred.size
print('Prediction accuracy on validation set is: ', accuracy)

with open("results.csv", "a") as myfile:
    finalstring = str(accuracy) + ";" + str(total_time) + ";" + str(model_type) + ";" + str(max_epoch) + ";" + str(
        args.iteration_number) + ";" + str(args.n_test) + ";" + str(args.n_train) + ";" + str(
        args.max_words) + ";" + str(args.min_words) + '\n'
    myfile.write(finalstring)