# IPGPF

Code for EMNLP 2023 long paper: An Iteratively Parallel Generation Method with the Pre-Filling Strategy for Document-level Event Extraction

### Data Preparation

ChFinAnn Link: (https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip)

DuEE-fin Link: (https://aistudio.baidu.com/aistudio/competition/detail/46)


1. Download the data from the link above.
```bash
cd IPGPF
mkdir ./Data 
```
2. For ChFinAnn data, unzip it to the `./Data` dictionary.
3. For DuEE-fin data, unzip it to the `./Data` dictionary.
4. Preprocess the DuEE-fin data to the same format as ChFinAnn data:
```bash
cd dee
python3 build_duee_data.py
```

### Training
```bash
# For a machine with 8 GPUs
# ChFinAnn dataset
$ bash train_chfinann.sh 8
# DuEE-fin dataset
$ bash train_duee.sh 8
```


Before that, make sure that you have set the correct dataset flag at line 369 of dee/event_type.py:
```python
dataset = ['ChFinAnn', 'DuEE-fin'][1]
```



Regarding the control variable settings, 
```bash
# ChFinAnn dataset
$ bash train_chfinann_CV.sh 8
# DuEE-fin dataset
$ bash train_duee_CV.sh 8
```

Before that, make sure that you have set the correct dataset flag at line 369 and entity merge flag at line 370 of dee/event_type.py.
```python
dataset = ['ChFinAnn', 'DuEE-fin'][1]
merge_entity = [True, False][1]
```


Please note that
- By setting a large step length of gradient accumulation, we can achieve large batch training with a few common GPUs.
Specifically, for Tesla V100 (32GB Memory), you should maintain `B/(N*G) == 1`,
where `B`, `N` and `G` denote the batch size, the number of GPUs, and the step size of gradient accumulation, respectively.

### Inference

To get inference results for DuEE-fin dataset, choose the best checkpoint on dev set and run
```bash
$ bash duee_inference.sh
```

### Citation

If you find our work useful to your research, you can cite the paper below:
```bash
@inproceedings{huang2023iteratively,
  title={An iteratively parallel generation method with the pre-filling strategy for document-level event extraction},
  author={Huang, Guanhua and Xu, Runxin and Zeng, Ying and Chen, Jiaze and Yang, Zhouwang and Weinan, E},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={10834--10852},
  year={2023}
}
```
