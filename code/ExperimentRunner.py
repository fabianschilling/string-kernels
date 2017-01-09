import numpy as np
from sklearn import svm, datasets
from sklearn import metrics
import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import sys
sys.path.append('../code/')
import wk
import ngk
import ssk
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt

class ExperimentRunner:
   
    EARN = 0
    ACQ = 1
    CRUDE = 2
    CORN = 3
    
    def prepare_data(self):
        #import data

        print "importing train + test data..."
        all_files = reuters.fileids()
        #print(len(all_files))

        train_files = list(filter(lambda file: file.startswith('train'), all_files))
        #print(len(train_files))

        test_files = list(filter(lambda file: file.startswith('test'), all_files))
        #print(len(test_files))

        categories = reuters.categories()

        earn_docs = reuters.fileids('earn')
        acq_docs = reuters.fileids('acq')
        crude_docs = reuters.fileids('crude')
        corn_docs = reuters.fileids('corn')

        earn_train_docs = [w for w in earn_docs if w in train_files]   
        earn_test_docs = [w for w in earn_docs if w in test_files]  

        acq_train_docs = [w for w in acq_docs if w in train_files]   
        acq_test_docs = [w for w in acq_docs if w in test_files]  

        crude_train_docs = [w for w in crude_docs if w in train_files]   
        crude_test_docs = [w for w in crude_docs if w in test_files]  

        corn_train_docs = [w for w in corn_docs if w in train_files]   
        corn_test_docs = [w for w in corn_docs if w in test_files]  


        #TODO: figure out how to pick out the docs we want
        for docHandle in earn_train_docs[0:10]:
            self.TrainDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TrainDocLabels.append(ExperimentRunner.EARN)

        for docHandle in acq_train_docs[0:10]:
            self.TrainDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TrainDocLabels.append(ExperimentRunner.ACQ)

        for docHandle in crude_train_docs[0:10]:
            self.TrainDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TrainDocLabels.append(ExperimentRunner.CRUDE)

        for docHandle in corn_train_docs[0:10]:
            self.TrainDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TrainDocLabels.append(ExperimentRunner.CORN)

        for docHandle in earn_test_docs[0:10]:
            self.TestDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TestDocLabels.append(ExperimentRunner.EARN)

        for docHandle in acq_test_docs[0:10]:
            self.TestDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TestDocLabels.append(ExperimentRunner.ACQ)

        for docHandle in crude_test_docs[0:10]:
            self.TestDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TestDocLabels.append(ExperimentRunner.CRUDE)

        for docHandle in corn_test_docs[0:10]:
            self.TestDocVals.append(" ".join(word_tokenize(reuters.raw(docHandle))))
            self.TestDocLabels.append(ExperimentRunner.CORN)

        print "done"
        
    def __init__(self):
        self.TrainDocVals = []
        self.TrainDocLabels = []
        self.TestDocVals = []
        self.TestDocLabels = []
    
        self.prepare_data()
        
        self.WKTestGram = np.ones((len(self.TestDocVals),len(self.TestDocVals)))
        self.WKTrainGram = np.ones((len(self.TrainDocVals),len(self.TrainDocVals)))

        self.NGKTestGram = np.ones((len(self.TestDocVals),len(self.TestDocVals)))
        self.NGKTrainGram = np.ones((len(self.TrainDocVals),len(self.TrainDocVals)))

        self.SSKTestGram = np.ones((len(self.TestDocVals),len(self.TestDocVals)))
        self.SSKTrainGram = np.ones((len(self.TrainDocVals),len(self.TrainDocVals)))

    def compute_gram_matrices(self,k,lamb):
    
        print "computing Gram matrices"

        for i in xrange( 0, len(self.TrainDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Train row %d of %d\n" % ( i+1, len(self.TrainDocVals) )       
            for j in xrange(0,len(self.TrainDocVals)):
                self.WKTrainGram[i][j] = wk.wk(self.TrainDocVals[i], self.TrainDocVals[j])
                #NOT YET READY
                #self.NGKTrainGram[i][j] = ngk.ngk(TrainDocVals[i], TrainDocVals[j], k)
                #self.SSKTrainGram[i][j] = ngk.ngk(TrainDocVals[i], TrainDocVals[j], k, lamb)
        for i in xrange( 0, len(self.TestDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Test row %d of %d\n" % ( i+1, len(self.TestDocVals) )       
            for j in xrange(0,len(self.TestDocVals)):
                self.WKTestGram[i][j] = wk.wk(self.TestDocVals[i], self.TestDocVals[j])
                #NOT YET READY
                #self.NGKTestGram[i][j] = ngk.ngk(TestDocVals[i], TestDocVals[j], k)
                #self.SSKTestGram[i][j] = ngk.ngk(TestDocVals[i], TestDocVals[j], k, lamb)

        print "done"

    def do_classification(self, trainKernel, testKernel, Ktype):
        #TODO: run 10 times like they do in the paper?
        print "classifying for %s" %Ktype
        clf = svm.SVC(kernel='precomputed')

        clf.fit(trainKernel, self.TrainDocLabels)
        label_pred = clf.predict(testKernel)

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.TestDocLabels, label_pred)

        #show table with results
        
        columns = ('F1 Score', 'Precision', 'Recall')
        rows = ['Earn', 'ACQ', 'Crude', 'Corn']

        cell_text = []

        cell_text.append([fscore[ExperimentRunner.EARN],precision[ExperimentRunner.EARN],recall[ExperimentRunner.EARN]])
        cell_text.append([fscore[ExperimentRunner.ACQ],precision[ExperimentRunner.ACQ],recall[ExperimentRunner.ACQ]])
        cell_text.append([fscore[ExperimentRunner.CRUDE],precision[ExperimentRunner.CRUDE],recall[ExperimentRunner.CRUDE]])
        cell_text.append([fscore[ExperimentRunner.CORN],precision[ExperimentRunner.CORN],recall[ExperimentRunner.CORN]])

        fig=plt.figure()
        ax = plt.gca()
        the_table = ax.table(cellText=cell_text, colLabels=columns, rowLabels=rows, loc='center')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.title(Ktype)

        plt.show()
        
    def run_test(self,k,lamb):
        self.compute_gram_matrices(k,lamb)
        self.do_classification(self.WKTrainGram, self.WKTestGram, 'WK')
        #do_classification(NGKTrainGram, NGKTestGram, 'NGK')
        #do_classification(SSKTrainGram, SSKTestGram, 'SSK')
