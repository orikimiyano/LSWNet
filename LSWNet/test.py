import os
import warnings
from data import *
from models.LSWNet import *
warnings.filterwarnings("ignore")

NameOfModel = 'LSWNet_1'

def test(case_path,case_name):
    testGene = testGenerator(case_path)
    lens = getFileNum(case_path)
    model = net(num_class=20)
    model.load_weights("saved_models/"+str(NameOfModel)+".hdf5")
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'/'+case_name,results)


for root, dirs, files in os.walk('data_1/test'):
    for dir in dirs:
        one_case_root = os.path.join(root, dir)
        test(one_case_root, dir)

