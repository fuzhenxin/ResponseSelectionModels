#encoding="utf-8"

import sys
import codecs
import pickle as pkl
import numpy as np

def get_voc(data_dir):

    file_name_voc = data_dir+"/voc.txt"
    file_name_voc = open(file_name_voc, "r")
    lines = [i.strip() for i in file_name_voc.readlines()]
    voc_dict = dict()
    for i,j in enumerate(lines):
        voc_dict[j]=i
    words_all = lines

    write_emb(words_all, data_dir)
    return voc_dict

def write_emb(words, data_dir):
    f = codecs.open(data_dir+"/emb.txt", "r", "utf-8", errors='ignore')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(len(lines))
    emb = dict()
    all_emb = []
    for line_index,line in enumerate(lines):
        line_w = line.split()[0]
        line_emb = np.array([float(i) for i in line.split()[1:]])
        if line_emb.shape[0]!=200:
            print("Error in: ", line_index)
            continue
        assert line_emb.shape[0]==200, (line_emb.shape[0], line_index)
        emb[line_w] = line_emb
        all_emb.append(line_emb)
    all_emb = np.array(all_emb)
    print(all_emb.shape)
    all_emb = np.mean(all_emb, axis=0)
    print("UNK emc shape: ", all_emb.shape)
    res = []
    for word in words:
        if word in emb:
            res.append(emb[word])
        else:
            res.append(all_emb)
    res = np.array(res)
    print("Glove shape: ", res.shape)
    pkl.dump(res, open(data_dir+"/emb.pkl", "wb"))


def convert_to_id(line, voc):
    #print("==============")
    #print(line)
    line = line.replace("\t", " | ")
    line = line.split()
    ids = []
    for word in line:
        if word == "|":
            word_id = -1
        else:
            word_id = voc[word] if word in voc else voc["<UNK>"]
        ids.append(word_id)
    if len(ids)==0:
        ids = [voc["<UNK>"]]
    return ids

def compose_data(data_dir, voc=None):
    res = []

    file_names = ["train.txt", "valid.txt", "test.txt"]

    for file_name_per in file_names:
        f = codecs.open(data_dir+"/"+file_name_per, "r", "utf-8")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        q_c, c_c, c_r, c_f, y = [], [], [], [], []
        for line in lines:
            line = line.split("|")
            ids_c1 = convert_to_id(line[0], voc)
            ids_r = convert_to_id(line[1], voc)
            q_c.append(ids_c1)
            c_r.append(ids_r)
            y.append(int(line[-1]))
        tvt = {"q_c": q_c, "c_r": c_r, "y": y}
        res.append(tvt)

    pkl.dump(res, open(data_dir+"/data.pkl", "wb"))

if __name__=="__main__":
    data_dir = sys.argv[1]
    voc = get_voc(data_dir)

    compose_data(data_dir, voc)

