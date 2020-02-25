# Response Selection Models
This repo collect the models for response selection and merge them into one framework.

This repo contains the following models:
- SMN: Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots. [pdf](https://arxiv.org/abs/1612.01627)
- DAM:  Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network. [pdf](https://www.aclweb.org/anthology/P18-1103/)
- IOI: One Time of Interaction May Not Be Enough: Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues. [pdf](https://www.aclweb.org/anthology/P19-1001/)
- ESIM: Sequential Matching Model for End-to-end Multi-turn Response Selection. [pdf](https://arxiv.org/abs/1901.02609)
- MSN: Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots. [pdf](https://www.aclweb.org/anthology/D19-1011/)

Note: I am still working on reproduce the results of MSN. The results pf MSN is still quite low in this repo. Welcome to pull requests to optimize MSN in this repo.

The models are all copied from their official released code and the dataset are from [MSN](https://github.com/chunyuanY/Dialogue). 
The data processing and code structure is from [DAM](https://github.com/baidu/Dialogue).

## Dependency
- python 3
- tensorflow 1.10
- scipy
- sklearn

### Data Preparation
- The dataset is from the data released by MSN.
- The dataset can be downloaded from 
```
cd utils
python parse_data.py ../ResponseSelection/ubuntu_data ../data/ubuntu_write
python compose_data.py ../data/ubuntu_write
```
### Train Model
The configuration is located in `main.py`.
```
python main.py ubuntu train
```
### Prediction
```
CHECKPOINT_NAME=../output/**
python main.py ubuntu test CHECKPOINT_NAME
```

## Results
## Official results from [Leaderboards](https://github.com/JasonForJoy/Leaderboards-for-Multi-Turn-Response-Selection/blob/master/README.md):
#### Ubuntu Dialogue Corpus V1
| Model |  R_2@1  |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | ------- | -------- | -------- | -------- | 
| SMN   |  0.926  |  0.726   |  0.847   |  0.961   | 
| DAM   |  0.938  |  0.767   |  0.874   |  0.969   | 
| IoI   |  0.947  |  0.796   |  0.894   |  0.974   | 
| MSN   | -       |  0.800   |  0.899   |  0.978   | 
#### Douban Conversation Corpus
| Model |  MAP  |  MRR  |  P@1  |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | ----- | ----- | ----- | -------- | -------- | -------- |
| SMN   | 0.529 | 0.569 | 0.397 |  0.233   |  0.396   |  0.724   |
| DAM   | 0.550 | 0.601 | 0.427 |  0.254   |  0.410   |  0.757   |
| IoI   | 0.573 | 0.621 | 0.444 |  0.269   |  0.451   |  0.786   |
| MSN   | 0.587 | 0.632 | 0.470 |  0.295   |  0.452   |  0.788   | 
#### E-commerce Corpus
| Model |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | -------- | -------- | -------- |
| SMN   |  0.453   |  0.654   |  0.886   |
| IoI   |  0.563   |  0.768   |  0.950   |
| MSN   |  0.606   |  0.770   |  0.937   |

## The results of this repo:
#### Ubuntu Dialogue Corpus V1
| Model |  R_2@1  |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | ------- | -------- | -------- | -------- | 
| SMN   |  0.944  |  0.782   |  0.885   |   0.973  | 
| DAM   |  0.947  |  0.790   |  0.890   |   0.975  |  
| ESIM  |         |          |          |          | 
| IoI   |         |          |          |          | 
#### Douban Conversation Corpus
| Model |  MAP  |  MRR  |  P@1  |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | ----- | ----- | ----- | -------- | -------- | -------- |
| SMN   | 0.550 | 0.589 | 0.407 |  0.253   |  0.411   |  0.769   |
| DAM   |       |       |       |          |          |          |
| ESIM  |       |       |       |          |          |          |
| IoI   |       |       |       |          |          |          |
#### E-commerce Corpus
| Model |  R_10@1  |  R_10@2  |  R_10@5  |
| ----- | -------- | -------- | -------- |
| SMN   |  0.530   |  0.703   |  0.928   |
| DAM   |          |          |          |
| ESIM  |          |          |          |
| IoI   |          |          |          |


## TODO
1. Upgrade tendorflow to newer version
2. Optimizing MSN

## Reference
We refer theese papers and repos to build this repo:
- SMN Code: https://github.com/MarkWuNLP/MultiTurnResponseSelection
- DAM Code: https://github.com/baidu/Dialogue
- IOI Code: https://github.com/chongyangtao/IOI
- ESIM Code: https://github.com/alibaba/esim-response-selection