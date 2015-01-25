# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:43:18 2014

@author: m126
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:42:37 2014

@author: m126
"""
import issct
import issct.classifier.neuronal as classNeuronal
import issct.transformation as trafo
import numpy as np
import random as rnd
from issct.tools.driver import load as _load
from issct.tools.driver import save as _save
from issct.tools.driver import cross_validation
from matplotlib.mlab import find as find

from issct.unsupervised.neuronal import ConvLayer as ConvLayer
from issct.unsupervised.neuronal import LSTMLayer as LSTMLayer
import issct.classifier.neuronal as neuClass
from theano import shared, function

def trainModel(data, cvid, l1reg=1e-5, l2reg= 1e-5, learningRate=0.0001,nValid = 1, nTrain=1):
    batchSize   = 50
    nEpochs     = 50
        
    print('training classifier:')      
    params = [(800, [('l2', 1e-4)],'rectLinear'),         
              (LSTMLayer,400, [('l2', l2reg)]),
              (800, [('l2', 1e-4)],'rectLinear'),
              (LSTMLayer,400, [('l2', l2reg)]),
              (800, [('l2', 1e-4)],'rectLinear')]
      
    
    #model = learner.train(nEpochs=nEpochs, learningRate=learningRate, sentLab = c)
    #train 3 models and take the best one:
    models = []
    errors = []
    for i in xrange(nTrain):
        #randomly choose and exclude a validation speaker from training set:
        spkids      = [find(data.methalist['spkid']== s) for s in unique(data.get_metha('spkid'))]
        rnd.shuffle(spkids)
        trainSpks   = spkids[:-nValid]
        testSpks    = spkids[-nValid:]
        
        train       = data.subset(trainSpks, 'spkid')
        test        = data.subset(testSpks, 'spkid')
        print(train)
        print(test)
        
        #shuffling while leaving the sentences intact:
        sentences   = train.get_metha('path')
        sentIDS     = np.unique(sentences)
        pList       = []
        batches     = []
        permIdx     = np.random.permutation(len(sentIDS))
        for idx in permIdx:
            pos=find(sentences==sentIDS[idx])
            if sum(sentences==sentIDS[idx]) > 0:
                batches.append((min(pos), max(pos)))
           
            pList.extend(pos)
           
        print('using ' +str(len(sentIDS)))+' sentences'
        learner = classNeuronal.learner()
        learner.set_params(useMomentum = 0.5, layers=params, nExclude=1, classActivation = 'softmax', learnSchedule=['exp', 1e7], regularizers = [('l1', l1reg),('l2', l2reg)])            
        
        learner.create(train, 'emotion', test, permutation=pList, batches=batches)
        model, error = learner.train(nEpochs=nEpochs, learningRate=learningRate, batchSize=batchSize, denoising= 0, dropout=0)
        models.append(model)
        errors.append(error)
    _save(model,'/home/m126/cvalModel.pkl')
    return models[argmax(errors)] #give the best model of the worst validation set back
    
if __name__ == '__main__':      
    batchSize = 9

    print('load training dataset:')
    data = issct.data.loadmat('/home/m126/emodb_mf40.mat')
    data = data.subset([3],'set') #only take tub,lss,ims   
    data = data.subset([1], 'voiced')
        #mean energy normalization for each sentence:
    eNorm = trafo.energyNorm()
    data = eNorm.map(data)
#    data = data.subset([17,31,37,45], 'emotion') #only take gr,gw,nt,tr,wt
    X = data.X
#    dX = np.concatenate((np.zeros((1,X.shape[1])),np.diff(X, n=1, axis=0)), axis = 0)
#    #ddX = np.concatenate((np.zeros((2,X.shape[1])),np.diff(X, n=2, axis=0)), axis = 0)
#    X = np.concatenate((X, dX), axis = 1)
    
    dataset = issct.data(X)
    dataset.add_metha('spkid',data.get_metha('spkid'))
    em =data.get_metha('emotion')
    em = [e[0]+e[1] for e in em]
    dataset.add_metha('emotion',em)
    dataset.add_metha('set',data.get_metha('set'))
    dataset.add_metha('path',data.get_metha('path'))
    dataset.add_metha('voiced',data.get_metha('voiced'))
    dataset = dataset.subset([1,2,3,4,5,6], 'emotion') #only take gr,gw,nt,tr,wt,
    dataset2 = issct.data(dataset.X)   
    dataset2.add_metha('spkid',dataset.get_metha('spkid'))
    em =dataset.get_metha('emotion')
    dataset2.add_metha('emotion',em)
    dataset2.add_metha('set',dataset.get_metha('set'))
    dataset2.add_metha('path',dataset.get_metha('path'))
    dataset2.add_metha('voiced',dataset.get_metha('voiced'))
    
    print('do sampling test batches')
    #join and remove unvoiced parts of speech:
    batches, blocks = dataset2.join(nblock=batchSize, nfeed=int(batchSize/2), join=np.ravel, consistent = 'path' )    
    #batches = batches.subset([1],'voiced')
    data = []
    dataset = []
    print('do normalization trainset')   
    uznorm          = trafo.unitzero(batches)
    batches         = uznorm.map(batches)
    featLength      = batches.X.shape[1]
    print(batches.X.shape)

    rparam = 5e-4
#    trainSpks = list(np.asarray(unique(batches.get_metha('spkid'))[3:],dtype=int)-1)
#    testSpks = list(np.asarray(unique(batches.get_metha('spkid'))[1:2],dtype=int)-1)
#    
#    train = batches.subset([0,2,3,4,6,7,8,9],'spkid')
#    test = batches.subset([1],'spkid')
#    batches = []
#    model = trainModel(train, 0, learningRate=1e-3, l1reg=0, l2reg=rparam)
#    res = model.eval(test)
#    res.set_column_ids([0,1,2,3])
#    jres, blocks    = res.join(sets=res.get_metha('path'), join=lambda x: np.sum(x,axis=0))
##    
###    
# Do some parameter optimization:    
    print('do speaker independent cross-validation') 
    regParams = [5e-4]
#    
    for rparam in regParams:
        results         = cross_validation(batches, trainModel, 'spkid', subset=[0,1,2,3,4,5,6,7,8,9], nPool=1, l1reg=rparam, l2reg=rparam, learningRate=5e-6)
        rTUB            = results
        results.set_column_ids([0,1,2,3])
        jres, blocks    = rTUB.join(sets=rTUB.get_metha('path'), join=lambda x: np.sum(x,axis=0))
        _save(rTUB, '/home/m126/resRecSubsetEmos4tiny'+str(rparam)+'.pkl')
        _save(jres, '/home/m126/jresRecSubsetEmos4tiny'+str(rparam)+'.pkl')
