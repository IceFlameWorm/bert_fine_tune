# Bert Fine Tune

基于0.6.2版本的[pytorch_bert_pretrained](https://github.com/huggingface/pytorch-transformers)(目前已更名为pytorch-transformers)，开发并fine_tune了几版不同的文本语义相似度模型。

## Features
1. 包含原有0.6.2版本的所有模型和功能
2. 开发的语义相似度模型是表示型的(其实就是编码器是bert的siamese network)，并不是原生提供的交互式的模型，这么做的目的是方便离线生成句向量，加速在线匹配速度。
3. 包含cos，contractive和soft max三种不同的模型

## 模型安装
### 源文件安装
1. 克隆本仓库到本地
2. 在仓库根目录执行 `pip install .`
3. 运行`pip list | grep pytorch`, 输出`pytorch-pretrained-bert            0.6.2`表明安装成功。

### 数据集以及模型参数文件
后续提供下载连接

## 模型使用
0.6.2版本的pytorch_bert_pretrained具体用法请参考[pytorch_bert_pretrained使用](./README_origin.md)。

### 新模型使用示例
```
# from pytorch_bert_pretrained import BertForPairWiseClassification
# from pytorch_bert_pretrained import BertForPairWiseClassification2
from pytorch_bert_pretrained import BertForPointWiseClassification
from pytorch_pretrained_bert import BertTokenizer

FINE_TUNED_PATH = '/efs/fine_tune/lcqmc/pointwise_old/lcqmc_fine_tune_40_1_1e-5/'

model = BertForPointWiseClassification.from_pretrained(FINE_TUNED_PATH)
tokenizer = BertTokenizer.from_pretrained(FINE_TUNED_PATH)

def bert_sim(text1, text2):
    tokens1 = ['[CLS]'] + tokenizer.tokenize(text1) + ['[SEP]']
    tokens2 = ['[CLS]'] + tokenizer.tokenize(text2) + ['[SEP]']
    ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    segs1 = [0] * len(ids1)
    segs2 = [0] * len(ids2)
    tokens_tensor1 = torch.tensor([ids1])
    segments_tensor1 = torch.tensor([segs1])
    tokens_tensor2 = torch.tensor([ids2])
    segments_tensor2 = torch.tensor([segs2])
    model.eval()
    with torch.no_grad():
        logits, vec1, vec2 = model(tokens_tensor1, tokens_tensor2, segments_tensor1, segments_tensor2)
        probs = torch.softmax(logits, dim = -1)
    return probs[0, 1].item(), vec1, vec2

text1 = "我很高兴"
text2 = "我很开心"
pos_prob, vec1, vec2 = bert_sim(text1, text2)
```
更多用法请参考`fine_tune/notebooks`目录

### 新模型训练
请参考`examples`目录下的`run_classifieer_fine_tune*`等文件。