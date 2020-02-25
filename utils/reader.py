import pickle as pickle
import numpy as np
import math

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c1 = np.array(data['q_c'])
    r = np.array(data['c_r'])

    assert len(y) == len(c1) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'q_c': c1[p], 'c_r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail', process_context=False):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0]*length, 0 # hh

    if real_length <= length:
        if not isinstance(_list[0], list):
            if process_context:
                if cut_type=="tail": _list = [0]*(length - real_length) + _list
                else: _list = _list + [0]*(length - real_length)
            else:
                _list.extend([0]*(length - real_length))
        else:
            if process_context:
                if cut_type=="tail": _list = [[0] for jj in range(length - real_length)] + _list
                else: _list = _list + [[0] for jj in range(length - real_length)]
            else:
                _list.extend([[0] for jj in range(length - real_length)])
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length


def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns_batches1 = []
    _tt_turns_len_batches1 = []
    _every_turn_len_batches1 = []
    _response_batches = []
    _response_len_batches = []
    _label_batches = []


    batch_len = math.ceil(float(len(data['y']))/conf['batch_size'])
    for batch_index in range(batch_len):

        _turns1 = []
        _tt_turns_len1 = []
        _every_turn_len1 = []
        _response = []
        _response_len = []
        _label = []

        for i in range(conf['batch_size']):
            index = batch_index * conf['batch_size'] + i
            if index>=len(data['y']):
                break

            c1 = data['q_c'][index]
            r = data['c_r'][index]
            y = data['y'][index]

            turns1 = split_c(c1, conf['_EOS_'])
            assert len(turns1)
            nor_turns1, turn_len1 = normalize_length(turns1, conf['max_turn_num'], turn_cut_type, process_context=True)

            nor_turns_nor_c1 = []
            term_len1 = []
            for c in nor_turns1:
                nor_c, nor_c_len = normalize_length(c, conf['max_turn_len'], term_cut_type)
                nor_turns_nor_c1.append(nor_c)
                term_len1.append(nor_c_len)

            r = [int(i) for i in r]
            nor_r, r_len = normalize_length(r, conf['max_turn_len'], term_cut_type)

            _turns1.append(nor_turns_nor_c1)
            _every_turn_len1.append(term_len1)
            _tt_turns_len1.append(turn_len1)
            _response.append(nor_r)
            _response_len.append(r_len)
            _label.append(y)

        _turns_batches1.append(_turns1)
        _tt_turns_len_batches1.append(_tt_turns_len1)
        _every_turn_len_batches1.append(_every_turn_len1)
        _response_batches.append(_response)
        _response_len_batches.append(_response_len)
        _label_batches.append(_label)

    ans = {
        "turns1": _turns_batches1, "tt_turns_len1": _tt_turns_len_batches1, "every_turn_len1":_every_turn_len_batches1,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches
    }   

    return ans 

if __name__ == '__main__':
    conf = { 
        "batch_size": 5,
        "max_turn_num": 6, 
        "max_turn_len": 20, 
        "_EOS_": 1,
    }
    train, val, test, test_human = pickle.load(open('../data/data.pkl', 'rb'))
    print('load data success')
    
    train_batches = build_batches(train, conf)
    val_batches = build_batches(val, conf)
    test_batches = build_batches(test, conf)
    test_batches = build_batches(test_human, conf)
    print('build batches success')
    
    #pickle.dump([train_batches, val_batches, test_batches], open('../data/batches.pkl', 'wb'))
    #print('dump success')
