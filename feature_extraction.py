"""
author: jian jiang
email: jian.jiang@kcl.ac.uk
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm


class ChunksDataset(data.Dataset):
	"""
	torch dataset for the raw data
	:param
	data_path: path to the raw data file, each item is a chunk with text, duration, talk_type, participant_id
	max_seq_length: the pretrained model only supports maximal sequence length 128 for input. Longer inputs will be truncated
	"""

	def __init__(self, data_path, max_seq_length=128):
		self.raw_data = np.load(data_path, allow_pickle=True)
		self.max_seq_length = max_seq_length

	def __len__(self):
		return len(self.raw_data)

	def __getitem__(self, idx):
		# data_point['text'] = item[0]
		# data_point['duration'] = item[1]
		# data_point['talk_type'] = talk_type
		# data_point['participant_id'] = label_part2
		# data_point['minutes_counter'] = minutes_counter_part2
		data_point = self.raw_data[idx]
		text = data_point['text']
		duration = data_point['duration']
		talk_type = data_point['talk_type']
		participant_id = data_point['participant_id']
		minutes_counter = data_point['minutes_counter']
		video_name = data_point['video_name']
		# text, duration, talk_type, part_id = self.raw_data[idx]

		if len(text) > self.max_seq_length:
			text = text[:self.max_seq_length]
		return text, int(participant_id), (duration, talk_type, minutes_counter, video_name)


class EmbsDataset(data.Dataset):
	"""
	Torch Dataset for the embeddings
    data_path: path to the embedding file, each item is a text embedding
    with label
	"""

	def __init__(self, data_path):
		data = np.load(data_path)
		self.emb = data['emb']
		self.label = data['label']

	def __len__(self):
		return len(self.emb)

	def __getitem__(self, item):
		return self.emb[item], self.label[item]


def encode_data2features(data_path):
	"""
	encode the raw data to features and save features to file
	"""
	batch_size = 128
	mode = data_path.split('_')[0]
	ds = ChunksDataset(data_path)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

	loader = tqdm(loader)

	# pretrained multi-lingual BERT
	# link: https://github.com/UKPLab/sentence-transformers
	model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
	# sentences = ["This is an example sentence", "Each sentence is converted"]
	# embeddings = model.encode(sentences)

	all_embs = []
	all_part_ids = []
	all_duration = []
	all_talk_type = []
	all_minutes_counter = []
	all_video_name = []
	for i, (text, part_id, meta_info) in enumerate(loader):
		embeddings = model.encode(text, batch_size=batch_size)
		all_embs += embeddings.tolist()
		all_part_ids += part_id.tolist()
		duration, talk_type, minutes_counter, video_name = meta_info
		all_duration += duration.tolist()
		all_talk_type += list(talk_type)
		all_minutes_counter += minutes_counter.tolist()
		all_video_name += video_name.tolist()


	all_embs = np.array(all_embs)
	all_part_ids = np.array(all_part_ids)
	all_duration = np.array(all_duration)
	all_talk_type = np.array(all_talk_type)
	all_minutes_counter = np.array(all_minutes_counter)
	all_video_name = np.array(all_video_name)

	# duration = np.array([meta_info[0] for meta_info in ds])
	# talk_type = np.array([meta_info[1] for meta_info in ds])
	# minutes_counter = np.array([meta_info[2] for meta_info in ds])
	np.savez(f'embeddings/{mode}_embeddings.npz', emb=all_embs, label=all_part_ids, duration=all_duration, talk_type=all_talk_type, minutes_counter=all_minutes_counter, video_name=all_video_name)
	# np.savez(f'embeddings/{mode}_embeddings.npz', emb=all_embs, label=all_part_ids, duration=duration, talk_type=talk_type, minutes_counter=minutes_counter)

	# np.savez(mode + "_embeddings.npz", emb=all_embs, label=all_part_ids)


def main():
	for mode in ['training', 'test', 'valid']:
		raw_data_path = mode + "_raw_data.npy"
		encode_data2features(raw_data_path)
		data_path = "embeddings/" + mode + "_embeddings.npz"
		data = np.load(data_path)
		ds = EmbsDataset(data_path)
		unique_labels = np.unique(ds.label)
		print(f'{mode} embeddings: number of unique labels {len(unique_labels)}; labels {unique_labels}; ')
		k=1

main()
