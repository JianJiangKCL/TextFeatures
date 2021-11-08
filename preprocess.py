"""
author: jian jiang
email: jian.jiang@kcl.ac.uk
"""
import numpy as np
import pysrt
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

    from pathlib import Path
    for path in Path(dir_path).rglob('*.srt'):
        subs = pysrt.open(path)
        file_name = path.stem
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

        for chunk in chunks_part1:
            item = combining_subs(chunk)
            # talk_type is for checking only
            item.append(talk_type)
            item.append(label_part1)
            data[label_part1].append(item)
        for chunk in chunks_part2:
            item = combining_subs(chunk)
            item.append(talk_type)
            item.append(label_part2)
            data[label_part2].append(item)
    data_np = []
    for _, chunks in data.items():
        for chunk in chunks:
            data_np.append(chunk)
    data_np = np.array(data_np)
    np.save('raw_data', data_np)
    return data_np


def main():
    dir_path = 'training_data_transcripts'
    data = process_srt_files(dir_path)
    loaded_data = np.load('raw_data.npy')
    is_equal = np.array_equal(data, loaded_data)

    print(is_equal)


if __name__ == '__main__':
    ##################
    # Test a single srt file
    # subs = pysrt.open('043079_animals.srt')
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



