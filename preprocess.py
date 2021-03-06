"""
author: jian jiang
email: jian.jiang@kcl.ac.uk
"""
import numpy as np
import pysrt
from pathlib import Path
PART1 = 'PART.1: '
PART2 = 'PART.2: '
EPS = 10
MILLISECONDS = 1000
TIME_MAX = 60 * MILLISECONDS + EPS  # in milliseconds
TIME_MIN = 15 * MILLISECONDS   # in milliseconds
REPLACEMENT_LISTS = [PART1, PART2, '\n']


def divide_participants(subs):
    sub_part1 = []
    sub_part2 = []
    for sub in subs.data:
        if PART1 in sub.text:
            sub_part1.append(sub)
        if PART2 in sub.text:
            sub_part2.append(sub)
    return sub_part1, sub_part2


def combining_subs(subs):
    com_text = ''
    com_duration = 0
    for sub in subs:
        for replacement in REPLACEMENT_LISTS:
            sub.text = sub.text.replace(replacement, ' ')
        com_text += sub.text
        com_duration += sub.duration.ordinal
    return [com_text, com_duration]


def chunk_subs(subs):
    time_accu = 0
    chunks = []
    chunk = []
    for i, sub in enumerate(subs):
        time_accu += sub.duration.ordinal
        chunk.append(sub)
        if time_accu >= TIME_MAX:
            chunks.append(chunk)
            time_accu = 0
            chunk = []
        # consider the last chunk, if it is too short than discard it
        if i == len(subs) - 1:
            if time_accu > TIME_MIN:
                chunks.append(chunk)

    return chunks


def process_srt_files(dir_path):
    data = {}
    mode = dir_path.split('_')[0]

    for path in Path(dir_path).rglob('*.srt'):
        #  encoding= 'unicode_escape' is not right
        try:
            subs = pysrt.open(path, encoding='utf-8')
        except:
            print(f"cannot process {path}")
        # subs = pysrt.open(path, encoding='utf-8')

        file_name = path.stem
        video_name = int(file_name[:6])
        label_part1 = int(file_name[:3])
        label_part2 = int(file_name[3:6])
        talk_type = file_name[7:]

        if label_part1 not in data.keys():
            data[label_part1] = []
        if label_part2 not in data.keys():
            data[label_part2] = []

        sub_part1, sub_part2 = divide_participants(subs)
        chunks_part1 = chunk_subs(sub_part1)
        chunks_part2 = chunk_subs(sub_part2)
        minutes_counter_part1 = 0
        for chunk in chunks_part1:
            item = combining_subs(chunk)
            # talk_type is for checking only
            data_point = {}
            data_point['text'] = item[0]
            data_point['duration'] = item[1]
            data_point['talk_type'] = talk_type
            data_point['participant_id'] = label_part1
            data_point['minutes_counter'] = minutes_counter_part1
            data_point['video_name'] = video_name
            # item.append(talk_type)
            # item.append(label_part1)
            # item.append(f"minute{minutes_counter_part1}")
            minutes_counter_part1 += 1

            data[label_part1].append(data_point)

        minutes_counter_part2 = 0
        for chunk in chunks_part2:
            item = combining_subs(chunk)
            data_point = {}
            data_point['text'] = item[0]
            data_point['duration'] = item[1]
            data_point['talk_type'] = talk_type
            data_point['participant_id'] = label_part2
            data_point['minutes_counter'] = minutes_counter_part2
            data_point['video_name'] = video_name
            # item.append(talk_type)
            # item.append(label_part2)
            # item.append(f"minute{minutes_counter_part2}")
            minutes_counter_part2 += 1
            data[label_part2].append(data_point)
    # sort data by id
    items = data.items()
    data_sorted = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    data_np = []
    for _, chunks in data.items():
        for chunk in chunks:
            data_np.append(chunk)
    data_np = np.array(data_np)
    np.save(mode + '_raw_data', data_np)
    return data_np


def main():
    # dir_path = 'training_data_transcripts'
    dir_path = 'test_data_transcripts'
    # dir_path = 'valid_data_transcripts'
    results_dir = 'split_data'
    for dir_path in ['training_data_transcripts', 'test_data_transcripts', 'valid_data_transcripts']:
        data = process_srt_files(dir_path)
        file_name = dir_path.split('_')[0]
        loaded_data = np.load(file_name + '_raw_data.npy', allow_pickle=True)
        is_equal = np.array_equal(data, loaded_data)

        print(is_equal)

if __name__ == '__main__':
    ##################
    # Test a single srt file
    # subs = pysrt.open('043079_animals.srt')
    # subs = pysrt.open('008105_talk.srt')
    # sub_part1, sub_part2 = divide_participants(subs)
    # chunks_part1 = chunk_subs(sub_part1)
    # chunks_part2 = chunk_subs(sub_part2)
    # data_part1 = list(map(lambda x: combining_subs(x), chunks_part1))
    # data_part2 = list(map(lambda x: combining_subs(x), chunks_part2))
    # print(data_part1)
    # print(data_part2)
    ##################

    ##################
    # Test embeddings for a single srt files
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    # embs = model.encode(data_part1[0][0], show_progress_bar=True, device='cuda', batch_size=32, normalize_embeddings=False)
    ##################

    ##################
    main()



