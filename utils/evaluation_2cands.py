import sys

def evaluation_one_session(data):
    if data[0][0] > data[1][0]:
        return 1.0
    else:
        return 0.0


def evaluate(file_path):
    sum_acc = 0

    i = 0
    total_num = 0
    with open(file_path, 'r') as infile:
        for line in infile:
            if i % 2 == 0:
                data = []

            tokens = line.strip().split('\t')
            data.append((float(tokens[0]), int(tokens[1])))

            data_remove = [j[1] for j in data]
            if i % 2 == 1 and sum(data_remove)>0:
                total_num += 1
                acc = evaluation_one_session(data)
                sum_acc += acc

            i += 1

    return sum_acc/total_num


if __name__ == '__main__':
    result = evaluate(sys.argv[1])
    print("Accuracy {:01.4f}".format(result))

