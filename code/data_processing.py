import json
from tqdm import tqdm
from collections import defaultdict
import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='input data process')
parser.add_argument('--year', type=int, default=2015, help='train_test_year')
args = parser.parse_args()
print(args)


def format_data():
    write_path = '../data/bibtest_2015'
    # the dicts to map numbers to entities
    author = {}
    venue = {}
    paper = {}
    topics = {}

    author_paper_train = defaultdict(lambda: [])
    paper_author_train = defaultdict(lambda: [])
    paper_citation_train = defaultdict(lambda: [])
    venue_paper_train = defaultdict(lambda: [])
    paper_topic_train = defaultdict(lambda: [])
    topic_paper_train = defaultdict(lambda: [])

    author_paper_test = defaultdict(lambda: [])
    paper_author_test = defaultdict(lambda: [])
    paper_citation_test = defaultdict(lambda: [])
    venue_paper_test = defaultdict(lambda: [])
    paper_topic_test = defaultdict(lambda: [])
    topic_paper_test = defaultdict(lambda: [])

    paper_venue = {}

    """
    TRAIN
    """

    with open('../../../JASIST_Diffusion/DiffusionDataInput/dblp_baby/brief_dblp.jsonl', 'r', encoding='UTF-8') as f:
        for line in f:
            line = json.loads(line)
            if line.get('year', '') and line.get('venue', '') and line.get('authors', []):
                if line['year'] <= 2015 and line['venue']['id']:
                    p_id = paper.setdefault(line['id'], len(paper))
                    a_ids = [author.setdefault(a, len(author)) for a in line['authors']]
                    v_id = venue.setdefault(line['venue']['id'], len(venue))
                    t_ids = [topics.setdefault(fos, len(topics)) for fos in line.get("foses", [])]
                    paper_venue[p_id] = v_id
                    for a in a_ids:
                        author_paper_train[a].append(p_id)
                    paper_author_train[p_id] = a_ids
                    paper_topic_train[p_id] = t_ids
                    for t in t_ids:
                        topic_paper_train[t].append(p_id)
                    venue_paper_train[v_id].append(p_id)
                if line['year'] > 2015 and line['venue']['id']:
                    p_id = paper.setdefault(line['id'], len(paper))
                    a_ids = [author.setdefault(a, len(author)) for a in line['authors']]
                    v_id = venue.setdefault(line['venue']['id'], len(venue))
                    t_ids = [topics.setdefault(fos, len(topics)) for fos in line.get("foses", [])]
                    paper_venue[p_id] = v_id
                    for a in a_ids:
                        author_paper_test[a].append(p_id)
                    paper_author_test[p_id] = a_ids
                    paper_topic_test[p_id] = t_ids
                    for t in t_ids:
                        topic_paper_test[t].append(p_id)
                    venue_paper_test[v_id].append(p_id)

    with open('../../../JASIST_Diffusion/DiffusionDataInput/dblp_baby/brief_dblp.jsonl', 'r', encoding='UTF-8') as f:
        for line in f:
            line = json.loads(line)
            if line['year'] <= 2015 and line['venue']['id'] and line['id'] in paper.keys():
                paper_citation_train[paper[line['id']]] = [paper[r] for r in line.get('references', []) if r in paper.keys()]
            if line['year'] > 2015 and line['venue']['id'] and line['id'] in paper.keys():
                paper_citation_test[paper[line['id']]] = [paper[r] for r in line.get('references', []) if r in paper.keys()]


    f = open(write_path + '/' + 'p_v.txt', 'w+')
    for key, value in paper_venue.items():
        f.write(f"{key},{value}" + '\n')

    relation_f_train = [
        "a_p_list_train.txt",
        "p_a_list_train.txt",
        "t_p_list_train.txt",
        "p_t_list_train.txt",
        "p_p_cite_list_train.txt",
        "v_p_list_train.txt",
    ]
    for file_path, lines in zip(relation_f_train,
                                [
                                    author_paper_train,
                                    paper_author_train,
                                    topic_paper_train,
                                    paper_topic_train,
                                    paper_citation_train,
                                    venue_paper_train]):
        current_file = open(write_path + '/' + file_path, 'w+')
        for x, y in lines.items():
            if y:
                if isinstance(y, list):
                    current_file.write(f"{x}:{','.join([str(z) for z in y])}\n")
                else:
                    current_file.write(f"{x}:{y}\n")

    relation_f_test = [
        "a_p_list_test.txt",
        "p_a_list_test.txt",
        "t_p_list_test.txt",
        "p_t_list_test.txt",
        "p_p_cite_list_test.txt",
        "v_p_list_test.txt",
    ]
    for file_path, lines in zip(relation_f_test,
                                [author_paper_test, paper_author_test, topic_paper_test, paper_topic_test,
                                 paper_citation_test, venue_paper_test]):
        current_file = open(write_path + '/' + file_path, 'w+')
        for x, y in lines.items():
            if y:
                if isinstance(y, list):
                    current_file.write(f"{x}:{','.join([str(z) for z in y])}\n")
                else:
                    current_file.write(f"{x}:{y}\n")

    save_t_community = open(write_path + '/t_community.txt', 'w')
    t_community = json.loads(open('../../community.json', 'r').read())
    for t, value in t_community.items():
        t = topics[t.split('|')[1]]
        save_t_community.write(f'{t}, {value}\n')

    open(write_path + '/' + 'topic.json', 'w+').write(json.dumps(topics, indent=4))
    open(write_path + '/' + 'author.json', 'w+').write(json.dumps(author, indent=4))
    open(write_path + '/' + 'venue.json', 'w+').write(json.dumps(venue, indent=4))
    open(write_path + '/' + 'paper.json', 'w+').write(json.dumps(paper, indent=4))


if __name__ == '__main__':
    format_data()
