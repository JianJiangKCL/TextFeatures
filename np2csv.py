import pandas as pd
import numpy as np
for mode in ['training', 'test', 'valid']:
	# raw_data_path = mode + "_raw_data.npy"
	# encode_data2features(raw_data_path)
	data_path = "embeddings/" + mode + "_embeddings.npz"
	data = np.load(data_path)
	# [n_emb, d_emb]
	# emb = data['emb']
	# dic = {}
	df_emb = pd.DataFrame(data['emb'])
	# df = pd.DataFrame.from_records([{ 'embedding': data['emb']}])
	# not
	df_meta = pd.DataFrame({'Video': data['video_name'], 'ID_y': data['label'], 'session': data['talk_type'],  'minute': data['minutes_counter']})
	df_meta['minute'] = df_meta['minute'] + 1
	df_meta['session'] = df_meta['session'].str.capitalize()
	df = pd.concat([df_meta, df_emb], axis=1)
	df.to_csv(mode + "_embeddings.csv", index=False)