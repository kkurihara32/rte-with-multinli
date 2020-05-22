import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEQUENCE_LENGTH = 30


def main():
    model = load_model("multinli_model.h5")
    print(model.summary())

    # tokenizerの読み込み
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # 学習結果の読み込み
    model = load_model("multinli_model.h5")

    # 入力データの整形
    sequences1 = tokenizer.texts_to_sequences([input()])
    sequences2 = tokenizer.texts_to_sequences([input()])
    x1 = pad_sequences(sequences1, maxlen=SEQUENCE_LENGTH)
    x2 = pad_sequences(sequences2, maxlen=SEQUENCE_LENGTH)

    y = model.predict([x1, x2])

    y_result = np.argmax(y)
    if y_result == 0:
        print("neutral")
    elif y_result == 1:
        print("contradiction")
    elif y_result == 2:
        print("entailment")
    else:
        print("unknown")



if __name__ == "__main__":
    main()
