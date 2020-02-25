
import sys, os
import pickle as pkl

def parse_data(dir_name_r, dir_name_w):
    if not os.path.exists(dir_name_w):
        os.mkdir(dir_name_w)
    voc, emb = pkl.load(open(dir_name_r+"/vocab_and_embeddings.pkl", "rb"))
    voc_num = max(voc.values())+1
    assert voc_num==len(emb)
    voc_list = ["UNK" for i in range(voc_num)]
    for i,j in voc.items():
        voc_list[j] = i

    # Process and Write datas
    for data_type in ["train", "dev", "test"]:
        f = open(dir_name_r+"/"+data_type+".pkl", "rb")
        if data_type=="dev":
            data_type="valid"
        f_w = open(dir_name_w+"/"+data_type+".txt", "w")
        cs, rs, ls = pkl.load(f)
        for c,r,l in zip(cs, rs, ls):
            c = [[ voc_list[j] for j in i if j!=0] for i in c]
            c = [ " ".join(i) for i in c]
            c = "\t".join(c).strip()
            r = [ voc_list[i] for i in r if i!=0]
            r = " ".join(r)
            line_w = "|".join([c,r,str(l)])+"\n"
            f_w.write(line_w)

    # write Voc and Emb
    f = open(dir_name_w+"/voc.txt", "w")
    for i in voc_list:
        f.write(i+"\n")
    f = open(dir_name_w+"/emb.txt", "w")
    for i,j in zip(voc_list, emb):
        j = [str(jj) for jj in j]
        j = " ".join(j)
        f.write(i+" "+j+"\n")


if __name__=="__main__":
    dir_name_r = sys.argv[1]
    dir_name_w = sys.argv[2]
    parse_data(dir_name_r, dir_name_w)












