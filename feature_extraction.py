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
		self.raw_data = np.load(data_path)
		self.max_seq_length = max_seq_length

	def __len__(self):
		return len(self.raw_data)

	def __getitem__(self, idx):
		text, duration, talk_type, part_id = self.raw_data[idx]

		if len(text) > self.max_seq_length:
			text = text[:self.max_seq_length]
		return text, int(part_id)


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
	for i, (text, part_id) in enumerate(loader):
		embeddings = model.encode(text, batch_size=batch_size)
		all_embs += embeddings.tolist()
		all_part_ids += part_id.tolist()
	all_embs = np.array(all_embs)
	all_part_ids = np.array(all_part_ids)

	np.savez(mode + "_embeddings.npz", emb=all_embs, label=all_part_ids)


def main():
	for mode in ['training', 'test', 'valid']:
		# raw_data_path = mode + "_raw_data.npy"
		# encode_data2features(raw_data_path)
		data_path = "embeddings/" + mode + "_embeddings.npz"
		data = np.load(data_path)
		ds = EmbsDataset(data_path)
		unique_labels = np.unique(ds.label)
		print(f'{mode} embeddings: number of unique labels {len(unique_labels)}; labels {unique_labels}; ')
		k=1

main()
