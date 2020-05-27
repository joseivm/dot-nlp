from scipy.stats.stats import pearsonr
import csv
import numpy as np

output_path = 'output/DPT/2705DPT1991_1939_test_preds.csv'

dpt = {'data' : [], 'people': [], 'things': []}
idx_to_dpt = {0: 'data', 1: 'people', 2: 'things'}
attr = {'ged': [], 'ehf': [], 'fd': [], 'dcp': [], 'sts': []}
idx_to_attr = {0: 'ged', 1: 'ehf', 2: 'fd', 3: 'dcp', 4: 'sts'}


with open(output_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        print(row[3], row[4])
        attr_values = row[3].split("', '")
        attr_values[0] = attr_values[0][2:]
        attr_values[-1] = attr_values[0][:2]
        attr_values = [float(x) for x in attr_values]
        for i, a in enumerate(attr_values):
            attr[idx_to_attr[i]].append(a)
        for i, a in enumerate(row[4]):
            dpt[idx_to_dpt[i]].append(float(a))

for d in dpt:
    print(d)
    for a in attr:
        corr = pearsonr(np.array(dpt[d]), np.array(attr[a]))
        print(f"{corr[0]:.2f}")
