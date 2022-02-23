import os
from pathlib import Path
import numpy as np

# data = {}
def count_unique_labels(dir_path):
	data = []
	for path in Path(dir_path).rglob('*.srt'):
		file_name = path.stem
		label_part1 = int(file_name[:3])
		label_part2 = int(file_name[3:6])
		# if label_part1 not in data.keys():
		data.append(label_part1)
		data.append(label_part2)
	data = np.array(data)
	unique_labels = np.unique(data)
	return unique_labels

for dir_name in  ['training', 'valid', 'test']:
	dir_path = dir_name + '_data_transcripts'
	unique_labels = count_unique_labels(dir_path)
	print(f'{dir_name}, number of unique labels {len(unique_labels)}; labels {unique_labels}; ')


k=1