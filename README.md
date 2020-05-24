# rte-with-multinli

## 使用方法

1. トレーニングデータの作成

  `python multi_nli_with_w2v.py`

  実行でtokenizer.pickle, multinli_model.h5の作成

2. 予測の実行

   `python prediction.py`

   実行後

   第一引数：前提文

   第二引数：仮説分

   を入力すると

   "neutral", "contradiction" ,"entailment" のいずれかで出力