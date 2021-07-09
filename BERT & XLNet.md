# BERT
## BERTとは

* 「Bidirectional Encoder Representations from Transformers」の略。
* Googleによって2018年に発表された言語処理の事前学習モデル
  * 汎用的な言語モデルを実現できる
  * データが少なくても学習できる
  * 双方向性を持つ

## モデルの概要
![モデル画像](https://qiita-user-contents.imgix.net/https%3A%2F%2Fimgur.com%2F1ol4NHO.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=e51ac8d5c804ad835a5d5c50a13eb5dc)

## 事前学習(pre-training)とfine-tuning
##### 事前学習(pre-training)
* BERTの事前学習は主に2つの手法によって行われている
  * Masked Language Model (以下MLM)
  * Next Sentence Prediction (以下NSP)

##### **MLM**
* 文中のいくつかの単語を[MASK]というトークンに置き換える
* 80%を[MASK]に置き換える`My dog is hairy -> My dog is [MASK]`
* 10%をランダムの単語に置き換える`My dog is apple`
* 10%はそのまま`My dog is hairy` 
* [MASK]に当てはまる単語を予測する


以下Google Colabでのサンプル(予測部分のみ)
```
!pip install transformers
!apt install aptitude
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
!pip install mecab-python3==0.7
!pip install fugashi
!pip install ipadic
```

```
from transformers import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
tokenizer.tokenize('私は犬が好き。')
tokenizer.encode('私は犬が好き。')
```

```
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
model.save_pretrained('mybert.bin')
model = BertForMaskedLM.from_pretrained('mybert.bin')
```

```
ids = tokenizer.encode('私は[MASK]が好き。')
ids
mskpos = ids.index(tokenizer.mask_token_id)
mskpos
```

```
import torch
x = torch.LongTensor(ids).unsqueeze(0)
a = model(x)
b = torch.topk(a[0][0][mskpos],k=5)
ans = tokenizer.convert_ids_to_tokens(b[1])
ans
```

