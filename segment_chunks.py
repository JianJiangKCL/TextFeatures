import pysrt

PART1 = 'PART.1: '
PART2 = 'PART.2: '
EPS = 10
MILLISECONDS = 1000
TIME_MAX = 60 * MILLISECONDS + EPS  # in milliseconds
TIME_MIN = 15 * MILLISECONDS   # in milliseconds

def divide_participants(subs):
    sub_part1 = []
    sub_part2 = []
    for sub in subs.data:
        if PART1 in sub.text:
            sub_part1.append(sub)
        if PART2 in sub.text:
            sub_part2.append(sub)
    return sub_part1, sub_part2

# low-efficient, because need to iterate through all subs twice
# def divide_participants(subs, PART_ID):
# 	sub_part = []
# 	for sub in subs.data:
# 		if PART_ID in sub.text:
# 			sub_part.append(sub)
# 	return sub_part

def combining_subs(subs):
    com_text = ''
    com_duration = 0
    for sub in subs:
        com_text += sub.text.replace(PART1, ' ').replace(PART2, ' ').replace('\n', ' ')
        com_duration += sub.duration.ordinal
    return [com_text, com_duration]


def chunk_subs(subs):
    time_accu = 0
    chunks = []
    chunk = []
    # total_ordinal = 0
    for i, sub in enumerate(subs):
        time_accu += sub.duration.ordinal
        # total_ordinal += sub.duration.ordinal
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


# saving data
def main():
    subs = pysrt.open('subs.srt')
    sub_part1, sub_part2 = divide_participants(subs)
    chunk_part1 = chunk_subs(sub_part1)
    chunk_part2 = chunk_subs(sub_part2)
    print('PART.1:')
    for chunk in chunk_part1:
        print(combining_subs(chunk))
    print('PART.2:')
    for chunk in chunk_part2:
        print(combining_subs(chunk))

if __name__ == '__main__':
    ##################
    # Test a single srt file
    subs = pysrt.open('043079_animals.srt')
    sub_part1, sub_part2 = divide_participants(subs)
    chunks_part1 = chunk_subs(sub_part1)
    chunks_part2 = chunk_subs(sub_part2)
    data_part1 = list(map(lambda x: combining_subs(x), chunks_part1))
    data_part2 = list(map(lambda x: combining_subs(x), chunks_part2))
    print(data_part1)
    print(data_part2)
    ##################

    ##################
    # Test embeddings for a single srt files
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    # embs = model.encode(data_part1[0][0], show_progress_bar=True, device='cuda', batch_size=32, normalize_embeddings=False)
    ##################


    k = 1

