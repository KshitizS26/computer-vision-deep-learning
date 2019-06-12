x = [0,0,0,0,3,3,3,3,4,4,4,4]
super_j = []
for i in range(0, len(x), 4):
	j_list = []
	for j in range(i, i+4):
		j_list.append(x[j])
	super_j.append(j_list)
print(super_j)
