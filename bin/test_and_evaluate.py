import sys
import os
import time

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
import utils.evaluation_douban as eva_douban


def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    data_collections = pickle.load(open(conf["data_path"], 'rb'))
    print('finish loading data')


    file_names = ["train.txt", "valid.txt", "test.txt"]

    test_data = data_collections[file_names.index("test.txt")]

    score_test = "score.test"


    test_batches = reader.build_batches(test_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' %conf)


    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # from tensorflow.python import debug as tf_debug
    # sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    with tf.Session(graph=_graph) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        _model.init.run()
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        test_type = conf["train_type"]
        logits = _model.trainops[test_type]["logits"]

        score_file_path = conf['save_path'] + '/' + score_test
        score_file = open(score_file_path, 'w')

        sim_his_all, sim_fut_all = [], []
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'starting test')
        for batch_index in range(test_batch_num):
            feed = {
                _model.turns1: test_batches["turns1"][batch_index],
                _model.tt_turns_len1: test_batches["tt_turns_len1"][batch_index],
                _model.every_turn_len1: test_batches["every_turn_len1"][batch_index],
                _model.response: test_batches["response"][batch_index],
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index],
                _model.keep_rate: 1.0,
            }

            scores = sess.run(logits, feed_dict = feed)

            for i in range(len(scores)):
                score_file.write(
                    str(scores[i]) + '\t' + 
                    str(test_batches["label"][batch_index][i]) + '\n')

        score_file.close()
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish test')

        #write evaluation result
        if "douban" in conf["data_path"]:
            result = eva_douban.evaluate(score_file_path)
            format_str = "MAP: {:01.4f} MRR {:01.4f} P@1 {:01.4f} R@1 {:01.4f} R@2 {:01.4f} R@5 {:01.4f}"
        else:
            result = eva.evaluate(score_file_path)
            format_str = "MRR: {:01.4f} R2@1 {:01.4f} R@1 {:01.4f} R@2 {:01.4f} R@5 {:01.4f}"
        print(format_str.format(*result))
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish evaluation')
