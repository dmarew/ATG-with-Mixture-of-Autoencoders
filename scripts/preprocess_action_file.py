from common import *
import csv



if __name__ =='__main__':
    action_seq = []
    with open('../data/observations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
                action_seq.append(int(row[2]))
    unique_actions = list(set(action_seq))
    action_seq = np.array(action_seq)
    for index, action in enumerate(unique_actions):
        action_seq[action_seq==action] = index
    print(action_seq.shape)
