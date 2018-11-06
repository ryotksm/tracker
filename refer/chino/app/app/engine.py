import numpy as np

from datetime import datetime
import glob
import psutil
import os
import json
from multiprocessing import Process

from app.log import get_logger

from PIL import Image
import chainer
from chainer import links as L
from chainer.links import GoogLeNet
from chainer.initializers import uniform
from chainer.links.connection.linear import Linear

from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer
from chainer.optimizers import AdaGrad
from chainer.training import extension
from chainer.serializers import npz
from chainer.datasets import LabeledImageDataset

from chainer.links.model.vision.googlenet import prepare


class GoogLeNetWrapper(chainer.Chain):
    def __init__(self, labels, model='auto', train=True):
        super(GoogLeNetWrapper, self).__init__()

        self.labels = labels

        with self.init_scope():
            self.gn = GoogLeNet(model)
            self.gn.disable_update()
            kwargs = {'initialW': uniform.LeCunUniform(scale=1.0)}
            self.fc_final = Linear(1024, len(labels), **kwargs)

    def __call__(self, x, train=True):
        h = self.gn(x, layers=['pool5'])['pool5']
        return self.fc_final(h)


class PreprocessedLabeledImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, root, random=True, force_rgb=False):
        self.base = LabeledImageDataset(pairs, root)
        self.random = random
        self.force_rgb = force_rgb

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        return prepare(image), label


class LogExtension(extension.Extension):
    def __init__(self, callback):
        super(LogExtension, self).__init__()
        self._callback = callback

    def __call__(self, trainer):
        t = trainer.stop_trigger
        e = trainer.updater.epoch_detail
        self._callback(t, e)


class SnapshotExtension(extension.Extension):
    def __init__(self, callback):
        super(SnapshotExtension, self).__init__()
        self._callback = callback

    def __call__(self, trainer):
        model_path = os.path.join(Engine.MODEL_DIR, '%s.npz' % (datetime.now().strftime('%Y%m%d_%H%M%S')))
        print(model_path)

        self._save_model(trainer.updater.get_optimizer('main').target.predictor, model_path)
        self._callback(model_path)

    def _save_model(self, model, model_path):
        npz.save_npz(model_path, model)

        with open(os.path.splitext(model_path)[0] + '.labels', 'w') as f:
            f.write(','.join(model.labels))


class Engine(object):
    WAITING = 1
    TRAINING = 10
    TRAINED = 20

    WORKDIR = os.path.join(os.path.dirname(__file__), '..')
    MODEL_DIR = '/models'
    STATUS_FILE_PATH = '/status.json'
    PRETRAINED_MODEL_PATH = '/bvlc_googlenet.npz'
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    DATASET_DIR = '/photos'

    def __init__(self):
        if os.path.isfile(Engine.STATUS_FILE_PATH):
            os.remove(Engine.STATUS_FILE_PATH)

        if not os.path.isdir(Engine.MODEL_DIR):
            os.makedirs(Engine.MODEL_DIR)

        self._status = {}
        self._update_status()

    def _reset_status(self):
        model_path = self._find_latest_model()
        self._status.update({
            'latest_model': model_path,
            'progress': 0.0,
            'period': 0,
            'epoch': 0.0,
            'training_phase': Engine.TRAINED if model_path else Engine.WAITING})

        self._save_status()

    def _update_status(self):
        p = self.training_phase

        if os.path.isfile(Engine.STATUS_FILE_PATH):
            with open(Engine.STATUS_FILE_PATH, 'r') as o:
                try:
                    self._status = json.load(o)
                except json.decoder.JSONDecodeError:
                    pass
        else:
            self._status = {}

        if p == Engine.TRAINING:
            self._status['training_phase'] = p
            self._save_status()
        else:
            self._reset_status()

    @property
    def training_phase(self):
        model_path = self._find_latest_model()
        waiting_status = Engine.TRAINED if model_path else Engine.WAITING

        pid = self._status.get('pid', -1)

        if pid >= 0:
            try:
                p = psutil.Process(self._status['pid'])
                if p.status() in ['running', 'disk-sleep', 'sleeping']:
                    return Engine.TRAINING
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                self._status['pid'] = -1

        return waiting_status

    def _find_latest_model(self):
        model_paths = glob.glob('/models/*.npz')

        if len(model_paths) > 0:
            return max(glob.glob('/models/*.npz'))
        else:
            return None

    @property
    def labels(self):
        model_path = self._find_latest_model()
        with open(os.path.splitext(model_path)[0] + '.labels') as f:
            labels = f.read().strip().split(',')
            return labels

    @property
    def status(self):
        self._update_status()
        return self._status

    def _save_status(self):
        with open(Engine.STATUS_FILE_PATH, 'w') as o:
            o.write(json.dumps(self._status))

    def infer(self, image):
        if self.training_phase != Engine.TRAINED:
            return None
        else:
            answer = self._infer(image)
            return answer

    def _infer(self, image):
        image_arr = Image.open(image)
        input_image = np.array([prepare(image_arr)])
        model_path = self._find_latest_model()
        with open(os.path.splitext(model_path)[0] + '.labels') as f:
            labels = f.read().strip().split(',')

        model = GoogLeNetWrapper(labels, model=Engine.PRETRAINED_MODEL_PATH)
        npz.load_npz(model_path, model)
        answer_ind = model(input_image).data.argmax()
        return model.labels[answer_ind]

    def _log_callback(self, trigger, epoch):
        self._status.update({
            'pid': os.getpid(),
            'period': trigger.period,
            'progress': min(1.0, epoch / trigger.period),
            'epoch': epoch})
        self._save_status()

    def _snapshot_callback(self, model_path):
        self._status['latest_model'] = model_path

    def find_dataset(self):
        dataset = {}
        subdirs = self.list_subdirs()

        for subdir in subdirs:
            label = os.path.basename(subdir)
            dataset[label] = ['/static%s' % (f) for f in self._list_images(subdir)]

        return dataset

    def generate_dataset(self):
        pairs = []
        subdirs = self.list_subdirs()

        for i, subdir in enumerate(subdirs):
            paths = self._list_images(subdir)
            pairs.extend((path, i) for path in paths)

        return PreprocessedLabeledImageDataset(pairs, root='.')

    def train(self):
        get_logger().info('start training')
        ds = self.generate_dataset()
        p = Process(target=self._train, args=(ds,))
        p.start()
        self._status.update({
            'pid': p.pid,
            'training_phase': Engine.TRAINING})
        self._save_status()

        return True

    def _train(self, ds):
        try:
            # XXX: always from scratch?
            model = GoogLeNetWrapper(self.list_labels(), model=Engine.PRETRAINED_MODEL_PATH)
            classifier = L.Classifier(model)
            train_iter = SerialIterator(ds, 1)
            get_logger().info(ds)
            optimizer = AdaGrad(0.05)
            optimizer.setup(classifier)
            updater = StandardUpdater(train_iter, optimizer)
            classifier.predictor.gn.disable_update()
            trainer = Trainer(updater, (1, 'epoch'))
            trainer.extend(LogExtension(self._log_callback), trigger=(1, 'iteration'))
            trainer.extend(SnapshotExtension(self._snapshot_callback), trigger=(1, 'epoch'))
            trainer.run()
        except Exception as e:
            get_logger().error(e)

    def _list_images(self, folder):
        lines = []

        for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(Engine.SUPPORTED_EXTENSIONS):
                    lines.append(os.path.join(dirpath, filename))

        return lines

    def list_images(self, subdirs):
        ret = {}

        for subdir in subdirs:
            # Use the directory name as the label
            label_name = subdir
            label_name = os.path.basename(label_name)
            label_name = label_name.replace('_', ' ')
            if label_name.endswith('/'):
                # Remove trailing slash
                label_name = label_name[0:-1]

            lines = []

            for dirpath, dirnames, filenames in os.walk(os.path.join(Engine.DATASET_DIR, subdir), followlinks=True):
                for filename in filenames:
                    if filename.lower().endswith(Engine.SUPPORTED_EXTENSIONS):
                        lines.append(os.path.join(dirpath, filename))

            ret[label_name] = lines

        return ret

    def list_labels(self):
        return [os.path.basename(d) for d in self.list_subdirs()]

    def list_subdirs(self):
        subdirs = []
        for filename in os.listdir(Engine.DATASET_DIR):
            subdir = os.path.join(Engine.DATASET_DIR, filename)
            if os.path.isdir(subdir):
                subdirs.append(subdir)

        subdirs.sort()

        return subdirs

    def validate_subfolder(self):
        return len(self.list_subdirs()) >= 2

    def validate_folder(self):
        if not os.path.exists(Engine.DATASET_DIR):
            return False
        if not os.path.isdir(Engine.DATASET_DIR):
            return False
        if not os.access(Engine.DATASET_DIR, os.R_OK):
            return False
        return True
