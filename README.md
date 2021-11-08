#  Texture Features Extraction
Python Scripts for extract texture features given transcripts.
>The extractor BERT is from this the repository:
[sentence-transformer](https://github.com/UKPLab/sentence-transformers);

>the pretrained weights are from: 
[distiluse-base-multilingual-cased-v2](https://huggingface.co/distiluse-base-multilingual-cased-v2/model) according to the paper [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813). Advantages of this model according to the paper: Aligned feature space. 1) Vector spaces are aligned across languages, i.e., identical sentences in different languages are close;
    2) vector space properties in the original source language from the teacher model M are adopted and transferred to other languages.



# Platform
- python: 3.6+
- Pytorch: 1.7+

## Howto


1. download `training_data_transcripts` in the root directory

```shell

training_data_transcript/
├── animals_transcripts1_train
    ├── 
    ├── 
    ├── 025157
        |── 025157_animals.srt
    ...



```
2. use `process_srt_files` in `preprocess.py`: to generate the raw data `raw_data.npy` that contains the list of chunks. 
Each chunk contains text, duration, talk_type, participant_id. One can use/modify `ChunksDataset` to load the raw data.


3. just run `feature_extraction.py` to extract the features and obtain the `embeddings.npz`. Each item contains a feature embedding and a participant label. One can use/modify `EmbsDataset` to load the extracted features.

## Aurthor information
```
Jian.Jiang@kcl.ac.uk
```
