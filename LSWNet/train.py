from data import *
from models.LSWNet import *
import sys
import os

NameOfModel = 'LSWNet_1'

class Logger(object) :
    def __init__(self, filename = str(NameOfModel)+".log") :
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message) :
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) :
        pass

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')


aug_args = dict()

train_gene = trainGenerator(batch_size=1,aug_dict=aug_args,train_path='data_1/',
                        image_folder='MR2D',label_folder='Mask2D',
                        image_color_mode='rgb',label_color_mode='rgb',
                        image_save_prefix='image',label_save_prefix='label',
                        flag_multi_class=True,save_to_dir=None)

model = net(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/'+str(NameOfModel)+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

