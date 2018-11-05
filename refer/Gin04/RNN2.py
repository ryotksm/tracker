# http://www.ie110704.net/2017/06/20/chainer%E3%81%A7%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%80%81rnn%E3%80%81cnn%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF/

import time
import numpy as np
import pandas as pd
from sklearn import datasets
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
 
# ���f���N���X��`
 
class NN(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # �N���X�̏�����
        # :param in_size: ���͑w�̃T�C�Y
        # :param hidden_size: �B��w�̃T�C�Y
        # :param out_size: �o�͑w�̃T�C�Y
        super(NN, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.Linear(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x):
        # ���`�d�̌v�Z���s���֐�
        # :param x: ���͒l
        h = F.sigmoid(self.xh(x))
        h = F.sigmoid(self.hh(h))
        y = self.hy(h)
        return y
# �w�K
 
EPOCH_NUM = 100
HIDDEN_SIZE = 20
BATCH_SIZE = 20
 
# �f�[�^
N = 100
in_size = 4
out_size = 3
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris["data"], iris["target"]], columns= iris["feature_names"] + ["target"])
data = np.array(data.values)
dataset = []
for d in data:
    x = d[0:4]
    y = d[4]
    dataset.append((np.array(x, dtype="float32"), np.array(y, dtype="int32")))
N = len(dataset)
 
# ���f���̒�`
model = L.Classifier(NN(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size))
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# �w�K�J�n
print("Train")
train, test = chainer.datasets.split_dataset_random(dataset, N-50) # 100�����w�K�p�A50�����e�X�g�p
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(10, "epoch"))) # 10�G�|�b�N���ƂɃ��O�o��
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # �G�|�b�N�A�w�K�����A�e�X�g�����A�w�K���𗦁A�e�X�g���𗦁A�o�ߎ���
#trainer.extend(extensions.ProgressBar()) # �v���O���X�o�[�o��
trainer.run()

# �\��
 
print("Predict")
print("x\ty\tpredict")
idx = np.random.choice(N, 10)
for i in idx:
    x = dataset[i][0]
    y_ = np.argmax(model.predictor(x=x.reshape(1,len(x))).data)
    y = dataset[i][1]
    print(x, "\t", y, "\t", y_)

