import sys
import os
import time

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# configure

data_name = sys.argv[1]
if data_name=="ecommerce":
    data_pre = "data/ecommerce_write"
    vocab_size = 32994
elif data_name=="douban":
    data_pre = "data/douban_write"
    vocab_size = 156157
elif data_name=="ubuntu":
    data_pre = "data/ubuntu_write"
    vocab_size = 158469
else:
    assert False

output_pre = "output"

model_name = sys.argv[3]
if sys.argv[2]=="train":
    init_model = None
else:
    init_model = sys.argv[4]

conf = {
    "data_path": data_pre + "/data.pkl",
    "save_path": output_pre + "/results_"+data_name+"_cr_"+model_name,
    "word_emb_init": data_pre + "/emb.pkl",
    "init_model": init_model,
    "data_name": data_name,

    "train_type": "cr",
    "cr_model": model_name, # SMN DAM IOI ESIM MSN

    "rand_seed": None,
    "print_step": 1000,

    "decay_step": 10000,
    "decay_rate": 0.8,
    "learning_rate": 1e-4,
    "early_stop_count": 3,
    "keep_rate": 1.0,

    "vocab_size": vocab_size,
    "emb_size": 200,
    "batch_size": 40,

    "max_turn_num": 10,
    "max_turn_len": 50,

    "max_to_keep": 1,
    "num_scan_data": 20,
    "_EOS_": -1,
    "final_n_class": 1,

    # DAM parameter
    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,
    "stack_num": 5,
    "attention_type": "dot",

    # IoI parameter
    "ioi_layer_num": 7,
}


if sys.argv[2]=="train":
    model = net.Net(conf, is_train=True)
    train.train(conf, model)
else:
    #test and evaluation, init_model in conf should be set
    model = net.Net(conf, is_train=False)
    test.test(conf, model)
