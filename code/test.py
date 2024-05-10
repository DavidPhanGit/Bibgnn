t_community = {}
from collections import Counter
for line in open('../data/bibtest_2015/t_community.txt', 'r'):
    t_community[line.split(', ')[0]] = int(line.split(', ')[1])

for line in open('../data/bibtest_2015/het_neigh_train.txt', 'r'):
    n_list = line.rstrip('\n').split(':')[1].split(',')
    print(Counter(t_community[n[1:]] for n in n_list if n.startswith('t')))