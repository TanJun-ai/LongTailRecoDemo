
# origin_file = './tail_supp_7_pos.txt'
# save_file = './tail_train.txt'
# origin_file = './tail_query_7_pos.txt'
# save_file = 'tail_test.txt'
origin_file = './top_supp_35_pos.txt'
save_file = 'top_train.txt'
with open(save_file, "w") as fo:
	with open(origin_file) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				print(l[:36])
				fo.write(' '.join(l[:36]))
				fo.write('\n')