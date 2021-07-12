# BERT
## BERTとは

* 「Bidirectional Encoder Representations from Transformers」の略。
* Googleによって2018年に発表された言語処理の事前学習モデル
  * 汎用的な言語モデルを実現できる
  * データが少なくても学習できる
  * 双方向性を持つ
![画像](https://data-analytics.fun/wp-content/uploads/2020/04/bert3-1024x415.png)

## モデルの概要
![モデル画像](https://qiita-user-contents.imgix.net/https%3A%2F%2Fimgur.com%2F1ol4NHO.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=e51ac8d5c804ad835a5d5c50a13eb5dc)

## Transformer
##### transformerの概要
![transformer model](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F331449%2F2acaeae0-5eef-ef26-2335-4a6a148e7414.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=0cb79b920c64178dd98c78d032c2ff6f)

***解説はこちら*** https://data-analytics.fun/2020/04/01/understanding-transformer/

## 事前学習(pre-training)とfine-tuning
##### 事前学習(pre-training)
* BERTの事前学習は主に2つの手法によって行われている
  * Masked Language Model (以下MLM)
  * Next Sentence Prediction (以下NSP)

## MLM
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

## NSP
* 2つの文が意味的に続いているかを判定する
* 正しい時はIsNext、そうでない時はNotNextと判定する
```
入力：[CLS] the man went to [MASK] store [SEP] ／he bought a gallon [MASK] milk [SEP]
判定：IsNext
入力：[CLS] the man went to [MASK] store [SEP]／penguin [MASK] are flight #less birds [SEP]
判定：NotNext
```
以下サンプル
```
import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertForNextSentencePrediction
```

```
model = BertForNextSentencePrediction.from_pretrained('cl-tohoku/bert-base-japanese')

text1='巴は誰ですか？' #前文
text2='巴は美しい女性です。' #後文

text1_toks = ["[CLS]"] + tokenizer.tokenize(text1)+ ["[SEP]"]
text2_toks = tokenizer.tokenize(text2)+ ["[SEP]"]
text = text1_toks + text2_toks
segments_ids = [0] * len(text1_toks)+ [1] * len(text2_toks)
print(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(text)
print(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```

```
model.eval()

prediction = model(tokens_tensor, token_type_ids =segments_tensors)
prediction=prediction[0] # tuple to tensor
#print(predictions)
softmax = nn.Softmax(dim=1)
prediction_sm = softmax(prediction)  
print(prediction_sm)
print(prediction_sm[0][1].item())
```
textの文章を変更する

## fine-tuning
* 入力に近い層を残し出力に近い層を入れ替えて再度学習を行う
* 公開されている学習済みモデルを用いることができる
  * 汎用性が高い
* 少ないデータセットでも学習が可能

***サンプルはこちら*** https://qiita.com/sugulu_Ogawa_ISID/items/697bd03499c1de9cf082

## 公開されている学習済みモデル
* [京都大学](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese)
* [東北大学](https://github.com/cl-tohoku/bert-japanese)
* [Stockmarks](https://drive.google.com/drive/folders/1iDlmhGgJ54rkVBtZvgMlgbuNwtFQ50V-)
* [NICT](https://alaginrc.nict.go.jp/nict-bert/index.html)
* [Loboro](https://github.com/laboroai/Laboro-BERT-Japanese)


# XLNet
## XLNetとは

* Carnegie Mellon大学とGoogle Brainの研究チームによって2019年に発表されたBERTの問題点を改善したモデル

## BERTの問題点
* MLMの[MASK]はfine-tuningやテスト時には存在しない
* 複数の[MASK]があった場合、[MASK]の間にある依存関係を学習できない
> `I like [MASK], and I play [MASK] [MASK] every day.`
> 
> (sports, table, tennis)や(music, the, guitar)など複数の可能性がある

## XLNetでの改善
* MLMではなく、並べ替え言語モデル (permutation language modeling) を学習させる

> `“play”, “table”, “every”, “sports”, “,” -> “tennis”`や
>
> `“sports”, “I”, “day”, “I”, “and”, “every”, “.”, “play”, “tennis” -> “like”`
![XLNet](https://ai-scholar.tech/wp-content/uploads/2019/08/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2019-08-11-13.52.39-768x588.png) 

## Transformer-XL
* Transformerと比べて長い文章の依存関係を捉えられる

***解説はこちら*** https://data-analytics.fun/2020/04/11/understanding-transformer-xl/
## XLNetの成果
* RACEの20のタスク全てでBERT超え、うち18のタスクで最高精度を達成した
  * 中国の中学生~高校生向けの英語の読解テストから作られたデータセット

## 公開されている学習済みモデル
* [Stockmarks](https://drive.google.com/drive/folders/1eyuk1Mtf3nvIIx6lXBPEoymReQ34A0tE)


# 参考資料
* 論文
  * 「BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding」(https://arxiv.org/abs/1810.04805)
  * 「Attention Is All You Need」(https://arxiv.org/abs/1706.03762)
  * 「XLNet: Generalized Autoregressive Pretraining for Language Understanding」(https://arxiv.org/abs/1906.08237)
  * 「Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context」(https://arxiv.org/abs/1901.02860)

* 資料
  * [【論文解説】Transformerを理解する](https://data-analytics.fun/2020/04/01/understanding-transformer/)
  * [【実装解説】日本語版BERTでlivedoorニュース分類：Google Colaboratoryで（PyTorch）](https://qiita.com/sugulu_Ogawa_ISID/items/697bd03499c1de9cf082)
  * [Transformer-XLを理解する](https://data-analytics.fun/2020/04/11/understanding-transformer-xl/)
  * [BERTを超えた自然言語処理の最新モデル「XLNet」](https://ai-scholar.tech/articles/treatise/xlnet-ai-228)
  * [XLNetを理解する](https://data-analytics.fun/2020/05/06/understanding-xlnet/#toc15)
  * [BERTを超えた？　XLNet を実際に使ってみた](https://qiita.com/masaharu_/items/0f794b2d24c3f0789054)

* 本
  * 『BERT・XLNetに学ぶ、言語処理における事前学習』
  * 『PyTorch自然言語処理プログラミング』
