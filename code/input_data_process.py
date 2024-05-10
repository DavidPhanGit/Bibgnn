import six.moves.cPickle as pickle
import numpy as np
import argparse
import string
import re
import random
import math
from collections import Counter
from itertools import *
import networkx as nx
import json
from tqdm import tqdm
from gensim.models import Word2Vec

parser = argparse.ArgumentParser(description='input data process')
parser.add_argument('--data_path', type=str, default='../data/bibtest_2015/', help='path to data')

args = parser.parse_args()

rest_args = json.loads(open(args.data_path + 'config.json').read())

parser.add_argument('--model_path', type=str, default=f'../model_save/{rest_args["folder_name"]}',
                    help='path to save model')
parser.add_argument('--A_n', type=int, default=rest_args["A_n"],
                    help='number of author node')
parser.add_argument('--P_n', type=int, default=rest_args["P_n"],
                    help='number of paper node')
parser.add_argument('--V_n', type=int, default=rest_args["V_n"],
                    help='number of venue node')
parser.add_argument('--T_n', type=int, default=rest_args["T_n"],
                    help='number of topic node')
parser.add_argument('--train_community', type=str, default='../data/bibtest_2015/train_community.json',
                    help='train community file')
parser.add_argument('--test_community', type=str, default='../data/bibtest_2015/train_community.json',
                    help='test community file')
parser.add_argument('--C_n', type=int, default=5,
                    help='number of node class label')
parser.add_argument('--walk_n', type=int, default=10,
                    help='number of walk per root node')
parser.add_argument('--walk_L', type=int, default=30,
                    help='length of each walk')
parser.add_argument('--window', type=int, default=7,
                    help='window size for relation extraction')
parser.add_argument('--T_split', type=int, default=2012,
                    help='split time of train/test data')

args = parser.parse_args()
print(args)


class input_data(object):
    def __init__(self, args):
        self.args = args

        a_p_list_train = [[] for k in range(self.args.A_n)]
        a_p_list_test = [[] for k in range(self.args.A_n)]

        p_a_list_train = [[] for k in range(self.args.P_n)]
        p_a_list_test = [[] for k in range(self.args.P_n)]

        t_p_list_train = [[] for k in range(self.args.T_n)]
        t_p_list_test = [[] for k in range(self.args.A_n)]

        p_t_list_train = [[] for k in range(self.args.P_n)]
        p_t_list_test = [[] for k in range(self.args.P_n)]

        p_p_cite_list_train = [[] for k in range(self.args.P_n)]
        p_p_cite_list_test = [[] for k in range(self.args.P_n)]

        v_p_list_train = [[] for k in range(self.args.V_n)]

        relation_f = [
            "a_p_list_train.txt",
            "a_p_list_test.txt",

            "p_a_list_train.txt",
            "p_a_list_test.txt",

            "t_p_list_train.txt",
            "t_p_list_test.txt",

            "p_t_list_train.txt",
            "p_t_list_test.txt",

            "p_p_cite_list_train.txt",
            "p_p_cite_list_test.txt",

            "v_p_list_train.txt"
        ]

        # store academic relational data
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.args.data_path + f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if f_name == 'a_p_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        a_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
                elif f_name == 'a_p_list_test.txt':
                    for j in range(len(neigh_list_id)):
                        a_p_list_test[node_id].append('p' + str(neigh_list_id[j]))

                elif f_name == 'p_a_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        p_a_list_train[node_id].append('a' + str(neigh_list_id[j]))
                elif f_name == 'p_a_list_test.txt':
                    for j in range(len(neigh_list_id)):
                        p_a_list_test[node_id].append('a' + str(neigh_list_id[j]))


                elif f_name == 't_p_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        t_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
                elif f_name == 't_p_list_test.txt':
                    for j in range(len(neigh_list_id)):
                        t_p_list_test[node_id].append('p' + str(neigh_list_id[j]))


                elif f_name == 'p_t_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        p_t_list_train[node_id].append('t' + str(neigh_list_id[j]))
                elif f_name == 'p_t_list_test.txt':
                    for j in range(len(neigh_list_id)):
                        p_t_list_test[node_id].append('t' + str(neigh_list_id[j]))

                elif f_name == 'p_p_cite_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        p_p_cite_list_train[node_id].append('p' + str(neigh_list_id[j]))
                elif f_name == 'p_p_cite_list_test.txt':
                    for j in range(len(neigh_list_id)):
                        p_p_cite_list_test[node_id].append('p' + str(neigh_list_id[j]))

                else:
                    for j in range(len(neigh_list_id)):
                        v_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
            neigh_f.close()

        # print (p_a_list_train[0])

        # store paper venue
        p_v = [0] * self.args.P_n
        p_v_f = open(self.args.data_path + 'p_v.txt', "r")
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(',', line)[0])
            v_id = int(re.split(',', line)[1])
            p_v[p_id] = v_id
        p_v_f.close()

        # paper neighbor: author + citation + venue
        p_neigh_list_train = [[] for k in range(self.args.P_n)]
        for i in range(self.args.P_n):
            p_neigh_list_train[i] += p_a_list_train[i]
            p_neigh_list_train[i] += p_p_cite_list_train[i]
            p_neigh_list_train[i] += p_t_list_train[i]
            p_neigh_list_train[i].append('v' + str(p_v[i]))
        # print p_neigh_list_train[11846]

        self.a_p_list_train = a_p_list_train
        self.a_p_list_test = a_p_list_test
        self.p_a_list_train = p_a_list_train
        self.p_a_list_test = p_a_list_test

        self.t_p_list_train = t_p_list_train
        self.t_p_list_test = t_p_list_test
        self.p_t_list_train = p_t_list_train
        self.p_t_list_test = p_t_list_test

        self.p_p_cite_list_train = p_p_cite_list_train
        self.p_p_cite_list_test = p_p_cite_list_test
        self.p_neigh_list_train = p_neigh_list_train
        self.v_p_list_train = v_p_list_train
        self.p_v = p_v

        self.train_community = json.loads(open(args.train_community, 'r').read())
        self.test_community = json.loads(open(args.test_community, 'r').read())

    def gen_het_rand_walk(self):
        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "w")
        # print len(self.p_neigh_list_train)
        for i in range(self.args.walk_n):
            for j in range(self.args.A_n):
                if len(self.a_p_list_train[j]):
                    curNode = "a" + str(j)
                    het_walk_f.write(curNode + " ")
                    for l in range(self.args.walk_L - 1):
                        if curNode[0] == "a":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.a_p_list_train[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "p":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.p_neigh_list_train[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "v":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.v_p_list_train[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "t":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.t_p_list_train[curNode])
                            het_walk_f.write(curNode + " ")
                    het_walk_f.write("\n")
        het_walk_f.close()

    def gen_meta_rand_walk_APVPA(self):
        meta_walk_f = open(self.args.data_path + "meta_random_walk_APVPA_test.txt", "w")
        # print len(self.p_neigh_list_train)
        for i in range(self.args.walk_n):
            for j in range(self.args.A_n):
                if len(self.a_p_list_train[j]):
                    curNode = "a" + str(j)
                    preNode = "a" + str(j)
                    meta_walk_f.write(curNode + " ")
                    for l in range(self.args.walk_L - 1):
                        if curNode[0] == "a":
                            preNode = curNode
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.a_p_list_train[curNode])
                            meta_walk_f.write(curNode + " ")
                        elif curNode[0] == "t":
                            preNode = curNode
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.t_p_list_train[curNode])
                            meta_walk_f.write(curNode + " ")
                        elif curNode[0] == "p":
                            curNode = int(curNode[1:])
                            if preNode[0] == "a":
                                preNode = "p" + str(curNode)
                                curNode = "p" + str(self.p_v[curNode])
                                meta_walk_f.write(curNode + " ")
                            else:
                                preNode = "p" + str(curNode)
                                curNode = random.choice(self.p_neigh_list_train[curNode])
                                meta_walk_f.write(curNode + " ")
                        elif curNode[0] == "v":
                            preNode = curNode
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.v_p_list_train[curNode])
                            meta_walk_f.write(curNode + " ")
                    meta_walk_f.write("\n")
        meta_walk_f.close()

    def a_a_collaborate_train_test(self):
        a_a_list_train = [[] for k in range(self.args.A_n)]
        a_a_list_test = [[] for k in range(self.args.A_n)]
        p_a_list = [self.p_a_list_train, self.p_a_list_test]

        for t in range(len(p_a_list)):
            for i in range(len(p_a_list[t])):
                for j in range(len(p_a_list[t][i])):
                    for k in range(j + 1, len(p_a_list[t][i])):
                        if t == 0:
                            a_a_list_train[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
                            a_a_list_train[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
                        else:  # remove duplication in test and only consider existing authors
                            if len(a_a_list_train[int(p_a_list[t][i][j][1:])]) and len(
                                    a_a_list_train[int(p_a_list[t][i][k][1:])]):  # transductive case
                                if int(p_a_list[t][i][k][1:]) not in a_a_list_train[int(p_a_list[t][i][j][1:])]:
                                    a_a_list_test[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
                                if int(p_a_list[t][i][j][1:]) not in a_a_list_train[int(p_a_list[t][i][k][1:])]:
                                    a_a_list_test[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))

        # print (a_a_list_train[1])

        for i in range(self.args.A_n):
            a_a_list_train[i] = list(set(a_a_list_train[i]))
            a_a_list_test[i] = list(set(a_a_list_test[i]))

        a_a_list_train_f = open(args.data_path + "a_a_list_train.txt", "w")
        a_a_list_test_f = open(args.data_path + "a_a_list_test.txt", "w")
        a_a_list = [a_a_list_train, a_a_list_test]
        train_num = 0
        test_num = 0
        for t in range(len(a_a_list)):
            for i in range(len(a_a_list[t])):
                # print (i)
                if len(a_a_list[t][i]):
                    if t == 0:
                        for j in range(len(a_a_list[t][i])):
                            a_a_list_train_f.write("%d, %d, %d\n" % (i, a_a_list[t][i][j], 1))
                            node_n = random.randint(0, self.args.A_n - 1)
                            while node_n in a_a_list[t][i]:
                                node_n = random.randint(0, self.args.A_n - 1)
                            a_a_list_train_f.write("%d, %d, %d\n" % (i, node_n, 0))
                            train_num += 2
                    else:
                        for j in range(len(a_a_list[t][i])):
                            a_a_list_test_f.write("%d, %d, %d\n" % (i, a_a_list[t][i][j], 1))
                            node_n = random.randint(0, self.args.A_n - 1)
                            while node_n in a_a_list[t][i] or node_n in a_a_list_train[i] or len(
                                    a_a_list_train[i]) == 0:
                                node_n = random.randint(0, self.args.A_n - 1)
                            a_a_list_test_f.write("%d, %d, %d\n" % (i, node_n, 0))
                            test_num += 2
        a_a_list_train_f.close()
        a_a_list_test_f.close()

        print("a_a_train_num: " + str(train_num))
        print("a_a_test_num: " + str(test_num))

    def a_p_citation_train_test(self):
        p_time = [0] * args.P_n
        p_time_f = open(args.data_path + "p_time.txt", "r")
        for line in p_time_f:
            line = line.strip()
            p_id = int(re.split('\t', line)[0])
            time = int(re.split('\t', line)[1])
            p_time[p_id] = time + 2005
        p_time_f.close()

        a_p_cite_list_train = [[] for k in range(self.args.A_n)]
        a_p_cite_list_test = [[] for k in range(self.args.A_n)]
        a_p_list = [self.a_p_list_train, self.a_p_list_test]
        p_p_cite_list_train = self.p_p_cite_list_train
        p_p_cite_list_test = self.p_p_cite_list_test

        for t in range(len(a_p_list)):
            for i in range(len(a_p_list[t])):
                for j in range(len(a_p_list[t][i])):
                    if t == 0:
                        p_id = int(a_p_list[t][i][j][1:])
                        for k in range(len(p_p_cite_list_train[p_id])):
                            a_p_cite_list_train[i].append(int(p_p_cite_list_train[p_id][k][1:]))
                    else:  # remove duplication in test and only consider existing papers
                        if len(self.a_p_list_train[i]):  # tranductive inference
                            p_id = int(a_p_list[t][i][j][1:])
                            for k in range(len(p_p_cite_list_test[p_id])):
                                cite_index = int(p_p_cite_list_test[p_id][k][1:])
                                if p_time[cite_index] < args.T_split and (cite_index not in a_p_cite_list_train[i]):
                                    a_p_cite_list_test[i].append(cite_index)

        for i in range(self.args.A_n):
            a_p_cite_list_train[i] = list(set(a_p_cite_list_train[i]))
            a_p_cite_list_test[i] = list(set(a_p_cite_list_test[i]))

        test_count = 0
        # print (a_p_cite_list_test[56])
        a_p_cite_list_train_f = open(args.data_path + "a_p_cite_list_train.txt", "w")
        a_p_cite_list_test_f = open(args.data_path + "a_p_cite_list_test.txt", "w")
        a_p_cite_list = [a_p_cite_list_train, a_p_cite_list_test]
        train_num = 0
        test_num = 0
        for t in range(len(a_p_cite_list)):
            for i in range(len(a_p_cite_list[t])):
                # print (i)
                # if len(a_p_cite_list[t][i]):
                if t == 0:
                    for j in range(len(a_p_cite_list[t][i])):
                        a_p_cite_list_train_f.write("%d, %d, %d\n" % (i, a_p_cite_list[t][i][j], 1))
                        node_n = random.randint(0, self.args.P_n - 1)
                        while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]:
                            node_n = random.randint(0, self.args.P_n - 1)
                        a_p_cite_list_train_f.write("%d, %d, %d\n" % (i, node_n, 0))
                        train_num += 2
                else:
                    for j in range(len(a_p_cite_list[t][i])):
                        a_p_cite_list_test_f.write("%d, %d, %d\n" % (i, a_p_cite_list[t][i][j], 1))
                        node_n = random.randint(0, self.args.P_n - 1)
                        while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]:
                            node_n = random.randint(0, self.args.P_n - 1)
                        a_p_cite_list_test_f.write("%d, %d, %d\n" % (i, node_n, 0))
                        test_num += 2
        a_p_cite_list_train_f.close()
        a_p_cite_list_test_f.close()

        print("a_p_cite_train_num: " + str(train_num))
        print("a_p_cite_test_num: " + str(test_num))

    def a_v_train_test(self):
        a_v_list_train = [[] for k in range(self.args.A_n)]
        a_v_list_test = [[] for k in range(self.args.A_n)]
        a_p_list = [self.a_p_list_train, self.a_p_list_test]
        for t in range(len(a_p_list)):
            for i in range(len(a_p_list[t])):
                for j in range(len(a_p_list[t][i])):
                    p_id = int(a_p_list[t][i][j][1:])
                    if t == 0:
                        a_v_list_train[i].append(self.p_v[p_id])
                    else:
                        if self.p_v[p_id] not in a_v_list_train[i] and len(a_v_list_train[i]):
                            a_v_list_test[i].append(self.p_v[p_id])

        for k in range(self.args.A_n):
            a_v_list_train[k] = list(set(a_v_list_train[k]))
            a_v_list_test[k] = list(set(a_v_list_test[k]))

        a_v_list_train_f = open(args.data_path + "a_v_list_train.txt", "w")
        a_v_list_test_f = open(args.data_path + "a_v_list_test.txt", "w")
        a_v_list = [a_v_list_train, a_v_list_test]
        # train_num = 0
        # test_num = 0
        # test_a_num = 0
        for t in range(len(a_v_list)):
            for i in range(len(a_v_list[t])):
                if t == 0:
                    if len(a_v_list[t][i]):
                        a_v_list_train_f.write(str(i) + ":")
                        for j in range(len(a_v_list[t][i])):
                            a_v_list_train_f.write(str(a_v_list[t][i][j]) + ",")
                        # train_num += 1
                        a_v_list_train_f.write("\n")
                else:
                    if len(a_v_list[t][i]):
                        # test_a_num += 1
                        a_v_list_test_f.write(str(i) + ":")
                        for j in range(len(a_v_list[t][i])):
                            a_v_list_test_f.write(str(a_v_list[t][i][j]) + ",")
                        # test_num += 1
                        a_v_list_test_f.write("\n")
        a_v_list_train_f.close()
        a_v_list_test_f.close()

    # print("a_v_train_num: " + str(train_num))
    # print("a_v_test_num: " + str(test_num))
    # print (float(test_num) / test_a_num)

    def a_t_write_train_test(self):
        train_num = 0
        test_num = 0
        write_path = '../data/bibtest_2015'
        f1 = open(args.data_path + "a_t_list_train.txt", "w")
        f2 = open(args.data_path + "a_t_list_test.txt", "w")

        authors = json.loads(open(write_path + '/' + 'author.json', 'r').read())
        topics = json.loads(open(write_path + '/' + 'topic.json', 'r').read())
        train_atg = nx.Graph(nx.jit_graph(
            open(f'../../../JASIST_Diffusion/input/dblp_baby_data/before_networks/AT.json', 'r',
                 encoding='UTF-8').read()))
        test_atg = json.loads(
            open(f'../../../JASIST_Diffusion/output/output_dblp_baby_timeline/weighted_ra.json').read())
        print(len(train_atg.edges))
        print(len(test_atg))
        valid_edges = [(line[0], line[1]) for line in test_atg if line[3]]

        for u, v in tqdm(train_atg.edges):
            if u in authors.keys() and v in topics.keys():
                f1.write("%d, %d, %d\n" % (authors[u], topics[v], 1))
                neg_topic = random.sample(train_atg.nodes, 1)[0]
                while train_atg.has_edge(u, neg_topic[0]) or neg_topic[0] not in topics.keys() or (
                        u, neg_topic[0]) in valid_edges:
                    neg_topic = random.sample(train_atg.nodes, 1)
                f1.write("%d, %d, %d\n" % (authors[u], topics[neg_topic[0]], 0))
                train_num += 2
            if v in authors.keys() and u in topics.keys():
                f1.write("%d, %d, %d\n" % (authors[v], topics[u], 1))
                neg_topic = random.sample(train_atg.nodes, 1)
                while train_atg.has_edge(v, neg_topic[0]) or neg_topic[0] not in topics.keys() or (
                        u, neg_topic[0]) in valid_edges:
                    neg_topic = random.sample(train_atg.nodes, 1)
                f1.write("%d, %d, %d\n" % (authors[v], topics[neg_topic[0]], 0))
                train_num += 2

        for line in test_atg:
            if line[1] in topics.keys() and line[0] in authors.keys():
                if line[3]:
                    f2.write("%d, %d, %d\n" % (authors[line[0]], topics[line[1]], 1))
                else:
                    f2.write("%d, %d, %d\n" % (authors[line[0]], topics[line[1]], 0))
                test_num += 1
        print("a_t_train_num: " + str(train_num))
        print("a_t_test_num: " + str(test_num))

    def t_t_recombination_test(self):
        t_t_list_train = [[] for k in range(self.args.T_n)]
        t_t_list_test = [[] for k in range(self.args.T_n)]
        p_t_list = [self.p_t_list_train, self.p_t_list_test]

        for t in range(len(p_t_list)):
            for i in range(len(p_t_list[t])):
                for j in range(len(p_t_list[t][i])):
                    for k in range(j + 1, len(p_t_list[t][i])):
                        if t == 0:
                            t_t_list_train[int(p_t_list[t][i][j][1:])].append(int(p_t_list[t][i][k][1:]))
                            t_t_list_train[int(p_t_list[t][i][k][1:])].append(int(p_t_list[t][i][j][1:]))
                        else:  # remove duplication in test and only consider existing authors
                            if len(t_t_list_train[int(p_t_list[t][i][j][1:])]) and len(
                                    t_t_list_train[int(p_t_list[t][i][k][1:])]):  # transductive case
                                if int(p_t_list[t][i][k][1:]) not in t_t_list_train[int(p_t_list[t][i][j][1:])]:
                                    t_t_list_test[int(p_t_list[t][i][j][1:])].append(int(p_t_list[t][i][k][1:]))
                                if int(p_t_list[t][i][j][1:]) not in t_t_list_train[int(p_t_list[t][i][k][1:])]:
                                    t_t_list_test[int(p_t_list[t][i][k][1:])].append(int(p_t_list[t][i][j][1:]))

        # print (t_t_list_train[1])

        for i in range(self.args.T_n):
            t_t_list_train[i] = list(set(t_t_list_train[i]))
            t_t_list_test[i] = list(set(t_t_list_test[i]))

        t_t_list_train_f = open(args.data_path + "t_t_list_train.txt", "w")
        t_t_list_test_f = open(args.data_path + "t_t_list_test.txt", "w")
        t_t_list = [t_t_list_train, t_t_list_test]
        train_num = 0
        test_num = 0
        for t in range(len(t_t_list)):
            for i in range(len(t_t_list[t])):
                # print (i)
                if len(t_t_list[t][i]):
                    if t == 0:
                        for j in range(len(t_t_list[t][i])):
                            if self.train_community[str(i)] == self.train_community[str(t_t_list[t][i][j])]:
                                t_t_list_train_f.write("%d, %d, %d\n" % (i, t_t_list[t][i][j], 1))
                            else:
                                t_t_list_train_f.write("%d, %d, %d\n" % (i, t_t_list[t][i][j], 2))
                            node_n = random.randint(0, self.args.T_n - 1)
                            while node_n in t_t_list[t][i] or len(t_t_list_train[node_n]) == 0:
                                node_n = random.randint(0, self.args.T_n - 1)
                            t_t_list_train_f.write("%d, %d, %d\n" % (i, node_n, 0))
                            train_num += 2
                    else:
                        for j in range(len(t_t_list[t][i])):
                            if self.test_community[str(i)] == self.test_community[str(t_t_list[t][i][j])]:
                                t_t_list_test_f.write("%d, %d, %d\n" % (i, t_t_list[t][i][j], 1))
                            else:
                                t_t_list_test_f.write("%d, %d, %d\n" % (i, t_t_list[t][i][j], 2))

                            node_n = random.randint(0, self.args.T_n - 1)
                            while node_n in t_t_list[t][i] or node_n in t_t_list_train[i] or len(
                                    t_t_list_train[node_n]) == 0:
                                node_n = random.randint(0, self.args.T_n - 1)
                            t_t_list_test_f.write("%d, %d, %d\n" % (i, node_n, 0))
                            test_num += 2
        t_t_list_train_f.close()
        t_t_list_test_f.close()

        print("t_t_train_num: " + str(train_num))
        print("t_t_test_num: " + str(test_num))

    def read_random_walk_corpus(self):
        walks = []
        inputfile = open(self.args.data_path + "het_random_walk.txt", "r")
        for line in inputfile:
            path = []
            node_list = re.split(' ', line)
            for i in range(len(node_list)):
                path.append(node_list[i])
            walks.append(path)
        inputfile.close()
        model = Word2Vec(walks, vector_size=128, window=5, min_count=0, workers=2, sg=1, hs=0, negative=5)
        print("Output...")
        # model.wv.save_word2vec_format("../data/node_embedding.txt")
        print(self.args.data_path + "node_net_embedding.txt")
        model.wv.save_word2vec_format(self.args.data_path + "node_net_embedding.txt")


input_data_class = input_data(args=args)

# input_data_class.a_t_write_train_test()
#
input_data_class.gen_het_rand_walk()

input_data_class.read_random_walk_corpus()

# input_data_class.gen_meta_rand_walk_APVPA()
#
#
# input_data_class.a_a_collaborate_train_test()  # set author-author collaboration data

input_data_class.t_t_recombination_test()  # set author-author collaboration data

#
#
# input_data_class.a_p_citation_train_test() #set author-paper citation data
#
#
# input_data_class.a_v_train_test()  # generate author-venue data
