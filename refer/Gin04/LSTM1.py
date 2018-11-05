# http://www.ie110704.net/2017/06/20/chainer%E3%81%A7%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%80%81rnn%E3%80%81cnn%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF/

import datetime
import numpy as np
import matplotlib.pylab as plt
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
 
# ���f���N���X��`
 
class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # �N���X�̏�����
        # :param in_size: ���͑w�̃T�C�Y
        # :param hidden_size: �B��w�̃T�C�Y
        # :param out_size: �o�͑w�̃T�C�Y
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh_x = L.Linear(hidden_size, 4 * hidden_size),
            hh_h = L.Linear(hidden_size, 4 * hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
        self.hidden_size = hidden_size
 
    def __call__(self, x, t=None, train=False):
        # ���`�d�̌v�Z���s���֐�
        # :param x: ���͒l
        # :param t: �����̗\���l
        # :param train: �w�K���ǂ���
        # :return: �v�Z�������� or �\���l
        if self.h is None:
            self.h = Variable(np.zeros((x.shape[0], self.hidden_size), dtype="float32"))
            self.c = Variable(np.zeros((x.shape[0], self.hidden_size), dtype="float32"))
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh_x(h) + self.hh_h(self.h)
        self.c, self.h = F.lstm(self.c, h)
        y = self.hy(self.h)
        if train:
            return F.mean_squared_error(y, t)
        else:
            return y.data
 
    def reset(self):
        # ���z�̏������ƃ������̏�����
        self.zerograds()
        self.h = None
        self.c = None
 
# �w�K
 
EPOCH_NUM = 1000
HIDDEN_SIZE = 5
BATCH_ROW_SIZE = 100 # �����������n��������~�j�o�b�`�Ɏ�荞�ނ�
BATCH_COL_SIZE = 100 # �~�j�o�b�`�ŕ������鎞�n��
 
# ���t�f�[�^
train_data = np.array([np.sin(i*2*np.pi/50) for i in range(50)]*10)
 
# ���t�f�[�^��ϊ�
train_x, train_t = [], []
for i in range(len(train_data)-1):
    train_x.append(train_data[i])
    train_t.append(train_data[i+1])
train_x = np.array(train_x, dtype="float32")
train_t = np.array(train_t, dtype="float32")
in_size = 1
out_size = 1
N = len(train_x)
 
# ���f���̒�`
model = LSTM(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# �w�K�J�n
print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
 
    # �~�j�o�b�`�w�K
    x, t = [], []
    #  �~�j�o�b�`�w�K�f�[�^�Ƃ��āA���n��S�̂���ABATCH_COL_SIZE���̎��n��𔲂��o�������̂��ABATCH_ROW_SIZE�p�ӂ���
    for i in range(BATCH_ROW_SIZE):
        index = np.random.randint(0, N-BATCH_COL_SIZE+1) # �����_���ȉӏ��A������BATCH_COL_SIZE��������������ꏊ����I��
        x.append(train_x[index:index+BATCH_COL_SIZE]) # BATCH_COL_SIZE���̎��n������o��
        t.append(train_t[index:index+BATCH_COL_SIZE])
    x = np.array(x, dtype="float32")
    t = np.array(t, dtype="float32")
    loss = 0
    total_loss = 0
    model.reset() # ���z�ƃ������̏�����
    for i in range(BATCH_COL_SIZE): # �e����������BATCH_ROW_SIZE���Ɠǂݍ���ő������v�Z����
        x_ = np.array([x[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] # ����i�̓��͒l
        t_ = np.array([t[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] # ����i+1�̒l�i�������̗\���l�j
        loss += model(x=x_, t=t_, train=True)
    loss.backward()
    loss.unchain_backward()
    total_loss += loss.data
    optimizer.update()
    if (epoch+1) % 100 == 0:
        ed = datetime.datetime.now()
        print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
        st = datetime.datetime.now()
 
# �\��
 
print("\nPredict")
predict = np.empty(0) # �\�����n��
inseq_size = 50
inseq = train_data[:inseq_size] # �\�����O�܂ł̎��n��
for _ in range(N - inseq_size):
    model.reset() # ��������������
    for i in inseq: # ���f���ɗ\�����O�܂ł̎��n���ǂݍ��܂���
        x = np.array([[i]], dtype="float32")
        y = model(x=x, train=False)
    predict = np.append(predict, y) # �Ō�̗\���l���L�^
    # ���f���ɓǂݍ��܂���\�����O���n���\���l�ōX�V����
    inseq = np.delete(inseq, 0)
    inseq = np.append(inseq, y)
 
plt.plot(range(N+1), train_data, color="red", label="t")
plt.plot(range(inseq_size+1, N+1), predict, color="blue", label="y")
plt.legend(loc="upper left")
plt.show()
