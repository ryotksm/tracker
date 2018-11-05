# http://www.ie110704.net/2017/06/20/chainer%E3%81%A7%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%80%81rnn%E3%80%81cnn%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF/

import time
import numpy as np
import pandas as pd
from sklearn import datasets
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
 
# モデルクラス定義
 
class NN(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(NN, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.Linear(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x, y=None, train=False):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return: 計算した損失 or 予測したラベル
        x = Variable(x)
        if train:
            y = Variable(y)
        h = F.sigmoid(self.xh(x))
        h = F.sigmoid(self.hh(h))
        y_ = F.softmax(self.hy(h))
        if train:
            loss, accuracy = F.softmax_cross_entropy(y_, y), F.accuracy(y_, y)
            return loss, accuracy
        else:
            return np.argmax(y_.data)
 
    def reset(self):
        # 勾配の初期化
        self.zerograds()
 
# 学習
 
EPOCH_NUM = 100
HIDDEN_SIZE = 20
BATCH_SIZE = 20
 
# データ
N = 100
in_size = 4
out_size = 3
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris["data"], iris["target"]], columns= iris["feature_names"] + ["target"])
data = np.array(data.values)
perm = np.random.permutation(len(data))
data = data[perm]
train, test = np.split(data, [N])
train_x, train_y, test_x, test_y = [], [], [], []
for t in train:
    train_x.append(t[0:4])
    train_y.append(t[4])
for t in test:
    test_x.append(t[0:4])
    test_y.append(t[4])
train_x = np.array(train_x, dtype="float32")
train_y = np.array(train_y, dtype="int32")
test_x = np.array(test_x, dtype="float32")
test_y = np.array(test_y, dtype="int32")
 
# モデルの定義
model = NN(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# 学習開始
print("Train")
st = time.time()
for epoch in range(EPOCH_NUM):
    # ミニバッチ学習
    perm = np.random.permutation(N) # ランダムな整数列リストを取得
    total_loss = 0
    total_accuracy = 0
    for i in range(0, N, BATCH_SIZE): 
        x = train_x[perm[i:i+BATCH_SIZE]]
        y = train_y[perm[i:i+BATCH_SIZE]]
        model.reset()
        loss, accuracy = model(x=x, y=y, train=True)
        loss.backward()
        loss.unchain_backward()
        total_loss += loss.data
        total_accuracy += accuracy.data
        optimizer.update()
    if (epoch+1) % 10 == 0:
        ed = time.time()
        print("epoch:\t{}\ttotal loss:\t{}\tmean accuracy:\t{}\ttime:\t{}".format(epoch+1, total_loss, total_accuracy/(N/BATCH_SIZE), ed-st))
        st = time.time()
 
# 予測

print("Predict")
res = []
for x, y in zip(test_x, test_y):
    y_ = model(x=x.reshape(1,len(x)), train=False)
    if y == y_:
        res.append(1)
    else:
        res.append(0)
accuracy = sum(res)/len(res)
print("test data accuracy: ", accuracy)
