import pickle

import pandas as pd
import numpy as np

from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


TEST_DATA_FILE = "multinli_1.0/multinli_1.0_dev_matched.txt"
# ボキャブラリー作成のためのファイル
TRAIN_DATA_FILE = "multinli_1.0/multinli_1.0_train.txt"
# ボキャブラリーの最大単語数
MAX_WARDS_NUM = 200000
# シーケンスの固定長
SEQUENCE_LENGTH = 30
# 訓練データのうち検証データに回すデータの割合
VALIDATION_RATE = 0.1
TEST_RATE = 0.1
# word2vecの学習済みモデルのファイルの選択（１単語を３００次元で表現しているモデル）
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin.gz"
# 1単語を300次元のベクトルで表している
EMBEDDING_DIM = 300

sample_word = "beautiful"

rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25


def get_label_index_labeling():
    return {"neutral": 0, "contradiction": 1, "entailment": 2, "-": 3}


def to_lower(text: str):
    return text.lower()


def load_data():
    train_df = pd.read_csv(TRAIN_DATA_FILE, sep="\t", usecols=["sentence1", "sentence2", "gold_label"], dtype={"sentence1": str, "sentence2": str, "gold_label": str})
    train_df.fillna("", inplace=True)
    sentence1 = train_df["sentence1"].apply(to_lower)
    sentence2 = train_df["sentence2"].apply(to_lower)
    y = train_df["gold_label"].map(get_label_index_labeling())
    tokenizer = Tokenizer(num_words=MAX_WARDS_NUM, oov_token="<UNK>")
    tokenizer.fit_on_texts(sentence1 + " " + sentence2)
    sequences1 = tokenizer.texts_to_sequences(sentence1)
    sequences2 = tokenizer.texts_to_sequences(sentence2)
    # pad_sequences: シーケンスを固定長にする作業
    # x1: 前提分, x2: 仮説文
    x1 = pad_sequences(sequences1, maxlen=SEQUENCE_LENGTH)
    x2 = pad_sequences(sequences2, maxlen=SEQUENCE_LENGTH)
    # permutation: 引数に入ったnp.array配列をシャッフル。引数がintのときはlist[range(5)]とかと同じ。
    perm = np.random.permutation(len(x1))

    # 訓練データと検証データの数の決定
    train_num = int(len(x1) * (1 - VALIDATION_RATE))
    train_index = perm[:train_num]
    valid_index = perm[train_num:]

    x1_train = x1[train_index]
    x2_train = x2[train_index]
    y_train = y[train_index]

    x1_valid = x1[valid_index]
    x2_valid = x2[valid_index]
    y_valid = y[valid_index]
    return \
        (x1_train, x2_train, y_train), (x1_valid, x2_valid, y_valid), tokenizer


def create_embedding_matrix(word_index: dict):
    words_num = min(MAX_WARDS_NUM, len(word_index)) + 1
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    # tokenizerの単語数 * embedding層のノード数 のゼロ行列を作成
    embedding_matrix = np.zeros([words_num, EMBEDDING_DIM])

    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    return embedding_matrix


def build_model(output_dim: int, embedding_matrix: np.ndarray, num_lstm=300):
    sequence1_input = Input(shape=(SEQUENCE_LENGTH, ), dtype="int32")
    sequence2_input = Input(shape=(SEQUENCE_LENGTH, ), dtype="int32")

    embed = StaticEmbedding(embedding_matrix)
    embedded_sequences1 = embed(sequence1_input)
    embedded_sequences2 = embed(sequence2_input)

    encode = LSTM(num_lstm, dropout=rate_drop_lstm,
                  recurrent_dropout=rate_drop_lstm)
    feat1 = encode(embedded_sequences1)
    feat2 = encode(embedded_sequences2)
    x = entail(feat1, feat2)
    preds = Dense(output_dim, activation="softmax")(x)
    model = Model(inputs=[sequence1_input, sequence2_input], outputs=preds)
    return model


def StaticEmbedding(embedding_matrix: np.array) -> Embedding:
    input_dim, output_dim = embedding_matrix.shape
    return Embedding(input_dim=input_dim,
                     output_dim=output_dim,
                     weights=[embedding_matrix],
                     input_length=SEQUENCE_LENGTH,
                     trainable=False
                     )


def entail(feat1, feat2, dense_num=300):
    x = concatenate([feat1, feat2])
    x = Dropout(rate_drop_dense)(x)
    x = BatchNormalization()(x)
    x = Dense(dense_num, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = BatchNormalization()(x)
    return x


def load_test_data(tokenizer: Tokenizer):
    train_df = pd.read_csv(TEST_DATA_FILE, sep="\t",
                           usecols=["sentence1", "sentence2", "gold_label"],
                           dtype={"sentence1": str, "sentence2": str,
                                  "gold_label": str})
    train_df.fillna("", inplace=True)
    sentence1 = train_df["sentence1"].apply(to_lower)
    sentence2 = train_df["sentence2"].apply(to_lower)
    y = train_df["gold_label"].map(get_label_index_labeling())
    sequences1 = tokenizer.texts_to_sequences(sentence1)
    sequences2 = tokenizer.texts_to_sequences(sentence2)
    # pad_sequences: シーケンスを固定長にする作業
    # x1: 前提分, x2: 仮説文
    x1 = pad_sequences(sequences1, maxlen=SEQUENCE_LENGTH)
    x2 = pad_sequences(sequences2, maxlen=SEQUENCE_LENGTH)
    # permutation: 引数に入ったnp.array配列をシャッフル。引数がintのときはlist[range(5)]とかと同じ。
    perm = np.random.permutation(len(x1))

    # 訓練データと検証データの数の決定
    test_num = int(len(x1) * TEST_RATE)
    test_index = perm[:test_num]

    x1_test = x1[test_index]
    x2_test = x2[test_index]
    y_test = y[test_index]

    return x1_test, x2_test, y_test


def run():
    class_num = len(get_label_index_labeling())
    (X1_train, X2_train, y_train), (X1_valid, X2_valid, y_valid), tokenizer =\
        load_data()

    # tokenizerの保存
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X1_test, X2_test, y_test = load_test_data(tokenizer)
    Y_train, Y_valid, Y_test = to_categorical(y_train, class_num), \
                               to_categorical(y_valid, class_num), \
                               to_categorical(y_test, class_num)
    # embedding_matrix: tokenizerのvocabularyをベクトル表現した行列
    embedding_matrix = create_embedding_matrix(tokenizer.word_index)
    # modelインスタンスの作成
    model = build_model(output_dim=class_num, embedding_matrix=embedding_matrix)
    model.compile(loss='categorical_crossentropy', optimizer='nadam',
                  metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    hist = model.fit([X1_train, X2_train], Y_train,
                     validation_data=([X1_valid, X2_valid], Y_valid),
                     epochs=10, batch_size=8192, shuffle=True,
                     callbacks=[early_stopping])
    score = model.evaluate([X1_test, X2_test], Y_test, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])

    # 学習結果の保存
    model.save("multinli_model.h5")

    # model_json_str = model.to_json()
    # open('multinli_model.json', 'w').write(model_json_str)
    # model.save_weights('multinli_weights.h5')


if __name__ == "__main__":
    run()

