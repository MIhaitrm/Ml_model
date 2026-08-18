#!/usr/bin/env python

import csv
import os
import glob

path = "npy_files"
# path = "n_file"
files = glob.glob(os.path.join(path, "*.npy"))
print(files)

# for f in files:
# with open("annotation_file.csv", "w") as file:
# 	writer = csv.writer(file)
# 	writer.writerow(files)

names = []
for file in files:
    name = os.path.basename(file)
    names.append(name)


with open('ann_file.csv', 'w') as f:
    for line in names:
        f.write(f"{line}\n")

