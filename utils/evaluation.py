import sys

def get_p_at_n_in_m(data, n, m, ind):
	pos_score = data[ind][0]
	curr = data[ind:ind+m]
	curr = sorted(curr, key = lambda x:x[0], reverse=True)

	if curr[n-1][0] <= pos_score:
		return 1
	return 0

def evaluate(file_path):
	#return 0., 0., 0., 0.
	data = []
	with open(file_path, 'r') as file:

		for line in file:
			line = line.strip()
			tokens = line.split()
		
			if len(tokens) != 2:
				continue
		
			data.append((float(tokens[0]), int(tokens[1])))
	
	print(len(data))
	assert len(data) % 10 == 0
	
	p_at_1_in_2 = 0.0
	p_at_1_in_10 = 0.0
	p_at_2_in_10 = 0.0
	p_at_5_in_10 = 0.0
	mrr = 0.0
	f_w = open(file_path+".sig", "w")

	length = int(len(data)/10)

	for i in range(0, length):
		ind = i * 10
		#print(i)
		assert data[ind][1] == 1, ind
	
		p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
		p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
		p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
		p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

		p_at_1_in_2_a = get_p_at_n_in_m(data, 1, 2, ind)
		p_at_1_in_10_a = get_p_at_n_in_m(data, 1, 10, ind)
		p_at_2_in_10_a = get_p_at_n_in_m(data, 2, 10, ind)
		p_at_5_in_10_a = get_p_at_n_in_m(data, 5, 10, ind)


		curr = data[ind: ind+10]
		curr = [j[0] for j in curr]
		max_pos = curr.index(max(curr))+1
		mrr += 1/max_pos

		f_w.write(" ".join(map(str, [p_at_1_in_2_a, p_at_1_in_10_a, p_at_2_in_10_a, p_at_5_in_10_a, 1/max_pos]))+"\n")


	return (mrr/length, p_at_1_in_2/length, p_at_1_in_10/length, p_at_2_in_10/length, p_at_5_in_10/length)
	

if __name__=="__main__":
	result = evaluate(sys.argv[1])
	print("MRR: {:01.4f} R2@1 {:01.4f} R@1 {:01.4f} R@2 {:01.4f} R@5 {:01.4f}".format(*result))