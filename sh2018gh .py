import numpy as np
import matplotlib.pylab as pl
import csv
import copy
import random
from pylab import *
import networkx as nx
import math
from numpy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
import numpy
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model

from scipy import interp
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
import warnings
warnings.filterwarnings("ignore")

interact = np.loadtxt('2018/ddi_mat.txt', dtype=int)
sim1 = np.loadtxt('2018/chem_sim.txt', dtype=float)
sim2 = np.loadtxt('2018/label_mat.txt', dtype=float)
sim3 = np.loadtxt('2018/off_sim.txt', dtype=float)

# sim1=np.loadtxt("sim_q.txt",dtype=float)
# sim2=np.loadtxt("sim_q.txt",dtype=float)
# sim3=np.loadtxt("sim_q.txt",dtype=float)
# interact=np.loadtxt("act_q.txt",dtype=int)
row, col = interact.shape
print(row,col)

def mat2vec(mat):
    return list(mat.reshape((mat.shape[0]*mat.shape[1])))

def cross_validation(drug_drug_matrix, CV_num, seed,id_):
    y_real = []
    y_proba = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    row,col=drug_drug_matrix.shape
    link_number=0
    link_position=[]
    index0=[]

    nonLinksPosition=[]
    for i in range(row):
        for j in range(col):
            if j > i:
                if drug_drug_matrix[i][j]==1:
                    link_position.append((i,j))
                    link_number+=1
                if drug_drug_matrix[i][j]==0:
                    nonLinksPosition.append((i,j))
                    index0.append(i*col+j)

    link_position = np.array(link_position)

    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)

    fold_num = link_number // CV_num
    print(fold_num)
    for CV in range(0,1):

        print('*********round:' + str(CV) + "**********\n")
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        test_index.sort()
        testLinkPosition = link_position[test_index]
        act= copy.deepcopy(drug_drug_matrix)
        for i in range(0, len(testLinkPosition)):
            act[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
            act[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
        testPosition = list(testLinkPosition) + list(nonLinksPosition)

        if id_==10:
           auc_score, fpr, tpr, aupr_score, ytest, W = our(act ,testPosition,index0)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==1:
            W = []
            f1 = open('svm2018val1.txt', 'r')
            h = f1.readline().split(',')
            for i in range(len(h) - 1):
                W.append(float(h[i]))
            f1.close()
            ytest = []
            f2 = open('y2018.txt', 'r')
            h = f2.readline().split(',')
            for i in range(len(h) - 1):
                ytest.append(float(h[i]))
            f2.close()
            fpr, tpr, auc_thresholds = roc_curve(ytest, W)

            fpr, tpr, ytest
            y_real.append(ytest)
            y_proba.append(W)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs = [0.9]

        if id_==2:
           auc_score, fpr, tpr, aupr_score, ytest, W = kncla(testPosition,index0)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==3:
           auc_score, fpr, tpr, aupr_score, ytest, W = lrcla(testPosition,index0)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==4:
           auc_score, fpr, tpr, aupr_score, ytest, W = nbcla(testPosition,index0)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==5:
           auc_score, fpr, tpr, aupr_score, ytest, W = dtcla(testPosition,index0)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)

        if id_==6:
           auc_score, fpr, tpr, aupr_score, ytest, W = Label_Propagation(act, testPosition)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)

        if id_==0:
           auc_score, fpr, tpr, aupr_score, ytest, W = vili(act, testPosition)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==7:
           auc_score, fpr, tpr, aupr_score, ytest, W = wei(act, testPosition)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==8:
           auc_score, fpr, tpr, aupr_score, ytest, W = c1(act, testPosition)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)
        if id_==9:
           auc_score, fpr, tpr, aupr_score, ytest, W = c2(act, testPosition)
           y_real.append(ytest)
           y_proba.append(W)
           tprs.append(interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           aucs.append(auc_score)


    return aucs, mean_fpr, tprs, y_real, y_proba

def Label_Propagation(act, testPosition):
    # sim = np.loadtxt('sim_q.txt',dtype=float)
    # row,col = sim.shape
    # interact = np.loadtxt("act_q.txt",dtype=int)

    interact = np.loadtxt('2018/ddi_mat.txt', dtype=int)
    sim = np.loadtxt('sim_2018.txt', dtype=float)

    row, col = sim.shape

    labale = np.array(mat2vec(interact))

    X_test = []
    # X_train = []
    for (a, b) in testPosition:
        X_test.append((a * col) + b)

    ytest = labale[X_test]

    predicted_probability = []
    alpha = 0.9
    similarity_matrix = np.matrix(sim)
    train_drug_drug_matrix = act
    D = np.diag(((similarity_matrix.sum(axis=1)).getA1()))
    N = LA.pinv(D) * similarity_matrix

    transform_matrix = (1 - alpha) * LA.pinv(np.identity(len(similarity_matrix)) - alpha * N)
    return_matrix = transform_matrix * train_drug_drug_matrix
    predict_matrix = return_matrix + np.transpose(return_matrix)

    for i in range(0, len(testPosition)):
        predicted_probability.append(predict_matrix[testPosition[i][0], testPosition[i][1]])
    predicted_probability = np.array(predicted_probability)

    precision, recall, pr_thresholds = precision_recall_curve(ytest, predicted_probability)
    aupr_score = auc(recall, precision)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, predicted_probability)

    auc_score = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[predicted_probability > threshold] = 1


    return auc_score, fpr, tpr, aupr_score, ytest, predicted_probability
def vili(act, testPosition):
    # sim = np.loadtxt('sim_q.txt',dtype=float)
    # row,col = sim.shape
    # interact = np.loadtxt("act_q.txt",dtype=int)

    interact = np.loadtxt('2018/ddi_mat.txt', dtype=int)
    sim = np.loadtxt('sim_2018.txt', dtype=float)

    row, col = sim.shape

    labale = np.array(mat2vec(interact))

    X_test = []
    # X_train = []
    for (a, b) in testPosition:
        X_test.append((a * col) + b)

    ytest = labale[X_test]
    # print("tytespos",type(testPosition))

    A = []
    r = np.zeros((sim.shape))
    for i in range(row):
        for j in range(col):
            for k in range(row):
                A.append(act[i, k] * sim[k, j])
            A.sort()
            A.reverse()
            r[i, j] = A[0]
            A = []

    np.fill_diagonal(r, 0)

    W = np.maximum(r, r.transpose())

    # prediction=np.zeros((sim.shape))

    n3 = []
    for (a, b) in testPosition:
        n3.append(W[a, b])
    W = np.array(n3)

    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)
    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    auc_score = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1


    return auc_score, fpr, tpr, aupr_score, ytest, W
def our(act,testPosition,index0):


    row, col = interact.shape

    def mat2vec(mat):
        return list(mat.reshape((mat.shape[0] * mat.shape[1])))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # calculate score1

    path2 = np.zeros(interact.shape)
    path2 = np.matmul(act, sim1) + np.matmul(sim1, act) + np.matmul(act, act)
    path3 = (np.matmul(np.matmul(act, sim1), act)) + (np.matmul(np.matmul(sim1, sim1), act)) + (
        np.matmul(np.matmul(act, sim1), sim1)) + (np.matmul(np.matmul(sim1, act), sim1)) + (
                np.matmul(np.matmul(act, act), act))
    s1 = path2 + path3
    np.fill_diagonal(s1, 0)
    max1 = 0
    for i in range(row):
        for j in range(col):
            if s1[i][j] > max1:
                max1 = s1[i][j]
    for i in range(row):
        for j in range(col):
            s1[i][j] = s1[i][j] / max1

    # calculate score2

    path2 = np.zeros(interact.shape)
    path2 = np.matmul(act, sim2) + np.matmul(sim2, act) + np.matmul(act, act)
    path3 = (np.matmul(np.matmul(act, sim2), act)) + (np.matmul(np.matmul(sim2, sim2), act)) + (
        np.matmul(np.matmul(act, sim2), sim2)) + (np.matmul(np.matmul(sim2, act), sim2)) + (
                np.matmul(np.matmul(act, act), act))
    s2 = path2 + path3
    np.fill_diagonal(s2, 0)
    max1 = 0
    for i in range(row):
        for j in range(col):
            if s2[i][j] > max1:
                max1 = s2[i][j]
    for i in range(row):
        for j in range(col):
            s2[i][j] = s2[i][j] / max1

    # calculate score3

    path2 = np.zeros(interact.shape)
    path2 = np.matmul(act, sim3) + np.matmul(sim3, act) + np.matmul(act, act)
    path3 = (np.matmul(np.matmul(act, sim3), act)) + (np.matmul(np.matmul(sim3, sim3), act)) + (
        np.matmul(np.matmul(act, sim3), sim3)) + (np.matmul(np.matmul(sim3, act), sim3)) + (
                np.matmul(np.matmul(act, act), act))
    s3 = path2 + path3
    np.fill_diagonal(s3, 0)
    max1 = 0
    for i in range(row):
        for j in range(col):
            if s3[i][j] > max1:
                max1 = s3[i][j]
    for i in range(row):
        for j in range(col):
            s3[i][j] = s3[i][j] / max1


    #     calculate sum Score

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(s1[i][j], 4), round(s2[i][j], 4), round(s3[i][j], 4) ]

                #
                list_score.append(A)
                A = []
    #

    #  calculate
    labale = np.array(mat2vec(interact))
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]

    #         random.shuffle(ytrain)
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(xtrain, ytrain);
    W = rf.predict_proba(xtest)[:, 1]


    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)
    aupr_score = auc(recall, precision)


    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1
    auc_score = auc(fpr, tpr)

    return auc_score, fpr, tpr, aupr_score, ytest, W
def svcla(testPosition,index0):
    from sklearn.svm import SVC


    # row, col = interact.shape
    labale = np.array(mat2vec(interact))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(sim1[i][j], 4), round(sim2[i][j], 4), round(sim3[i][j], 4) ]

                #
                list_score.append(A)


    #  calculate
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]

    clf = SVC(decision_function_shape='ovr', gamma='auto', kernel='sigmoid', probability=True)
    clf.fit(xtrain, ytrain)
    print(clf.fit(xtrain, ytrain))
    W = clf.predict_proba(xtest)[:, 1]
    print("w:", W[:10])

    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    roc_auc = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1

    f1 = f1_score(ytest, predicted_score)
    accura = accuracy_score(ytest, predicted_score)
    pr = precision_score(ytest, predicted_score)
    rc = recall_score(ytest, predicted_score)

    print('Precision', pr)
    print('Recall', rc)
    print('F1-measure', f1)
    print('area under crov', roc_auc)
    print('area under pr-rc', aupr_score)
    print("acu", accura)

    return roc_auc, fpr, tpr, aupr_score, ytest,W
def dtcla(testPosition, index0):



    row, col = interact.shape
    labale = np.array(mat2vec(interact))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(sim1[i][j], 4), round(sim2[i][j], 4), round(sim3[i][j], 4)]

                #
                list_score.append(A)

    #  calculate
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(xtrain,ytrain)
    W=clf.predict_proba(xtest)[:,1]

    print("w:", W[:10])
    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    roc_auc = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1

    f1 = f1_score(ytest, predicted_score)
    accura = accuracy_score(ytest, predicted_score)
    pr = precision_score(ytest, predicted_score)
    rc = recall_score(ytest, predicted_score)

    print('Precision', pr)
    print('Recall', rc)
    print('F1-measure', f1)
    print('area under crov', roc_auc)
    print('area under pr-rc', aupr_score)
    print("acu", accura)


    return roc_auc, fpr, tpr, aupr_score, ytest, W
def nbcla(testPosition, index0):

    row, col = interact.shape
    labale = np.array(mat2vec(interact))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(sim1[i][j], 4), round(sim2[i][j], 4), round(sim3[i][j], 4) ]

                #
                list_score.append(A)

    #  calculate
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]

    gnb = GaussianNB()
    gnb.fit(xtrain, ytrain)
    W = gnb.predict_proba(xtest)[:, 1]

    print("w:", W[:10])
    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    roc_auc = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1

    f1 = f1_score(ytest, predicted_score)
    accura = accuracy_score(ytest, predicted_score)
    pr = precision_score(ytest, predicted_score)
    rc = recall_score(ytest, predicted_score)

    print('Precision', pr)
    print('Recall', rc)
    print('F1-measure', f1)
    print('area under crov', roc_auc)
    print('area under pr-rc', aupr_score)
    print("acu", accura)


    return roc_auc, fpr, tpr, aupr_score, ytest, W
def kncla(testPosition, index0):


    row, col = interact.shape
    labale = np.array(mat2vec(interact))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(sim1[i][j], 4), round(sim2[i][j], 4), round(sim3[i][j], 4) ]

                #
                list_score.append(A)

    #  calculate
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]

    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier(n_neighbors=3)
    kn.fit(xtrain, ytrain)
    W = kn.predict_proba(xtest)[:, 1]

    print("w:", W[:10])
    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    roc_auc = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1

    f1 = f1_score(ytest, predicted_score)
    accura = accuracy_score(ytest, predicted_score)
    pr = precision_score(ytest, predicted_score)
    rc = recall_score(ytest, predicted_score)

    print('Precision', pr)
    print('Recall', rc)
    print('F1-measure', f1)
    print('area under crov', roc_auc)
    print('area under pr-rc', aupr_score)
    print("acu", accura)


    return roc_auc, fpr, tpr, aupr_score, ytest, W
def lrcla(testPosition, index0):


    row, col = interact.shape
    labale = np.array(mat2vec(interact))

    X_test = []
    X_train = []
    for (a, b) in testPosition:
        X_test.append(a * col + b)

    for i in range(row):
        for j in range(col):
            if j > i:
                if (i * col + j) not in X_test:
                    X_train.append((i * col + j))
    X_train = X_train + index0

    X_test.sort()
    X_train.sort()

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    X = []

    k = 0
    for i in range(row):
        for j in range(col):
            if j > i:
                X.append(((i, j), k))
                k += 1
    X = dict(X)

    x_train = []
    x_test = []
    for e in X_train:
        i = (e // col)
        j = e % col
        x_train.append(X[(i, j)])

    for e in X_test:
        i = int(e // col)
        j = e % col
        x_test.append(X[(i, j)])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    list_score = []
    for i in range(row):
        for j in range(col):
            if j > i:
                A = [round(sim1[i][j], 4), round(sim2[i][j], 4), round(sim3[i][j], 4) ]

                #
                list_score.append(A)

    #  calculate
    core = np.array(list_score)
    xtrain = core[x_train]
    xtest = core[x_test]
    ytrain = labale[X_train]
    ytest = labale[X_test]
    from sklearn import linear_model

    lrc = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    lrc.fit(xtrain, ytrain)
    W = lrc.predict_proba(xtest)[:, 1]
    print("w:", W[:10])


    precision, recall, pr_thresholds = precision_recall_curve(ytest, W)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(ytest, W)

    roc_auc = auc(fpr, tpr)

    predicted_score = np.zeros(len(ytest))
    predicted_score[W > threshold] = 1

    f1 = f1_score(ytest, predicted_score)
    accura = accuracy_score(ytest, predicted_score)
    pr = precision_score(ytest, predicted_score)
    rc = recall_score(ytest, predicted_score)

    print('Precision', pr)
    print('Recall', rc)
    print('F1-measure', f1)
    print('area under crov', roc_auc)
    print('area under pr-rc', aupr_score)
    print("acu", accura)


    return roc_auc, fpr, tpr, aupr_score, ytest, W
def wei(act,testPosition):
    weights, cf1, cf2 = internal_determine_parameter(copy.deepcopy(act))

    multiple_predict_matrix  = ensemble_method(copy.deepcopy(drug_drug_matrix), act, testPosition)

    auc_score, fpr, tpr, aupr_score, ytest, W= ensemble_scoringw(copy.deepcopy(drug_drug_matrix),
                                                                                    multiple_predict_matrix,
                                                                                    testPosition, weights )

    return auc_score, fpr, tpr, aupr_score, ytest, W
def c1(act,testPosition):
    weights, cf1, cf2 = internal_determine_parameter(copy.deepcopy(act))

    multiple_predict_matrix = ensemble_method(copy.deepcopy(drug_drug_matrix), act, testPosition)

    auc_score, fpr, tpr, aupr_score, ytest, W= ensemble_scoring1(copy.deepcopy(drug_drug_matrix),
                                                                                    multiple_predict_matrix,
                                                                                    testPosition,cf1 )

    return auc_score, fpr, tpr, aupr_score, ytest, W
def c2(act,testPosition):
    weights, cf1, cf2 = internal_determine_parameter(copy.deepcopy(act))

    multiple_predict_matrix  = ensemble_method(copy.deepcopy(drug_drug_matrix), act, testPosition)

    auc_score, fpr, tpr, aupr_score, ytest, W= ensemble_scoring2(copy.deepcopy(drug_drug_matrix),
                                                                                    multiple_predict_matrix,
                                                                                    testPosition,cf2 )

    return auc_score, fpr, tpr, aupr_score, ytest, W
class Topology:
    def topology_similarity_matrix(drug_drug_matrix):
       drug_drug_matrix=np.matrix(drug_drug_matrix)
       G = nx.from_numpy_matrix(drug_drug_matrix)
       drug_num=len(drug_drug_matrix)
       common_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       AA_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       RA_similarity_matrix=np.zeros(shape=(drug_num,drug_num))

       eigenValues,eigenVectors = linalg.eig(drug_drug_matrix)
       idx = eigenValues.argsort()[::-1]
       eigenValues = eigenValues[idx[0]]

       beta=0.5*(1/eigenValues)
       Katz_similarity_matrix=inv(np.identity(drug_num)-beta*drug_drug_matrix)-np.identity(drug_num)
       for i in range(0,drug_num):
         for j in range(i+1,drug_num):
             commonn_neighbor=list(nx.common_neighbors(G, i, j))
             common_similarity_matrix[i][j]=len(commonn_neighbor)
             AA_score=0
             RA_score=0
             for k in range(0,len(commonn_neighbor)):
                  AA_score=AA_score+1/math.log(len(list(G.neighbors(commonn_neighbor[k]))))
                  RA_score=RA_score+1/len(list(G.neighbors(commonn_neighbor[k])))
             AA_similarity_matrix[i][j]=AA_score
             RA_similarity_matrix[i][j]=RA_score

             common_similarity_matrix[j][i]=common_similarity_matrix[i][j]
             AA_similarity_matrix[j][i]=AA_similarity_matrix[i][j]
             RA_similarity_matrix[j][i]=RA_similarity_matrix[i][j]

       D=np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
       L=D-drug_drug_matrix
       LL=pinv(L)
       LL=np.matrix(LL)
       ACT_similarity_matrix=np.zeros(shape=(drug_num,drug_num))
       for i in range(0,drug_num):
          for j in range(i+1,drug_num):
              ACT_similarity_matrix[i][j]=1/(LL[i,i]+LL[j,j]-2*LL[i,j])
              ACT_similarity_matrix[j][i]=ACT_similarity_matrix[i][j]

       D=np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
       N=pinv(D)*drug_drug_matrix
       alpha=0.9
       RWR_similarity_matrix=(1-alpha)*pinv(np.identity(drug_num)-alpha*N)
       RWR_similarity_matrix= RWR_similarity_matrix+ np.transpose(RWR_similarity_matrix)

       return np.matrix(common_similarity_matrix),np.matrix(AA_similarity_matrix),np.matrix(RA_similarity_matrix),np.matrix(Katz_similarity_matrix),np.matrix(ACT_similarity_matrix),np.matrix(RWR_similarity_matrix)

def load_csv(filename,type): #  load csv, ignore the first row,type=int, data read as intï¼Œ else float
        matrix_data=[]
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row_vector in csvreader:
              if type=='int':
                 matrix_data.append(list(map(int,row_vector[1:])))
              else:
                 matrix_data.append(list(map(float,row_vector[1:])))
        return np.matrix(matrix_data)

def modelEvaluation(real_matrix,predict_matrix,testPosition,featurename): #  compute cross validation results
       real_labels=[]
       predicted_probability=[]

       for i in range(0,len(testPosition)):
           real_labels.append(real_matrix[testPosition[i][0],testPosition[i][1]])
           predicted_probability.append(predict_matrix[testPosition[i][0],testPosition[i][1]])

#       normalize=MinMaxScaler()
       #predicted_probability= normalize.fit_transform(predicted_probability)
       real_labels=np.array(real_labels)
       predicted_probability=np.array(predicted_probability)

       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)

       all_F_measure=np.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              all_F_measure[k]=0
       max_index=all_F_measure.argmax()
       threshold=pr_thresholds[max_index]

       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)
       predicted_score=np.zeros(len(real_labels))
       predicted_score[predicted_probability>threshold]=1

       f=f1_score(real_labels,predicted_score)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score)
       print('results for feature:'+featurename)
       print('************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f, f-measure:%.3f************************' %(auc_score,aupr_score,recall,precision,accuracy,f))
       auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
       results=[auc_score,aupr_score,precision, recall,accuracy,f]
       return results

def fitFunction(individual,parameter1,parameter2):
       real_labels=parameter1
       multiple_prediction=parameter2
       ensemble_prediction=np.zeros(len(real_labels))
       for i in range(0,len(multiple_prediction)):
            ensemble_prediction=ensemble_prediction+real(individual[i])*real(multiple_prediction[i])
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, ensemble_prediction)
       aupr_score = auc(recall, precision)
       return (aupr_score),

def getParamter(real_matrix, multiple_matrix, testPosition):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_float", random.uniform, 0, 1)
    # Structure initializers
    variable_num = len(multiple_matrix)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, variable_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #################################################################################################
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        #predicted_probability = normalize.fit_transform(predicted_probability)
        multiple_prediction.append(predicted_probability)

    #################################################################################################
    toolbox.register("evaluate", fitFunction, parameter1=real_labels, parameter2=multiple_prediction)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(0)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    pop.sort(key=lambda ind: ind.fitness, reverse=True)

    return pop[0]

class MethodHub():
    def neighbor_method(similarity_matrix,train_drug_drug_matrix):
        return_matrix=np.matrix(train_drug_drug_matrix)*np.matrix(similarity_matrix)
        D=np.diag(((similarity_matrix.sum(axis=1)).getA1()))
        return_matrix=return_matrix*LA.pinv(D)
        return_matrix=return_matrix+np.transpose(return_matrix)
        return return_matrix

    def Label_Propagation(similarity_matrix,train_drug_drug_matrix):
       alpha=0.9
       similarity_matrix=np.matrix(similarity_matrix)
       train_drug_drug_matrix=np.matrix(train_drug_drug_matrix)
       D=np.diag(((similarity_matrix.sum(axis=1)).getA1()))
       N=pinv(D)*similarity_matrix

       transform_matrix=(1-alpha)*LA.pinv(np.identity(len(similarity_matrix))-alpha*N)
       return_matrix=transform_matrix*train_drug_drug_matrix
       return_matrix=return_matrix+np.transpose(return_matrix)
       return return_matrix

    def generate_distrub_matrix(drug_drug_matrix):
       A=np.matrix(drug_drug_matrix)
       [num,num]=A.shape
       upper_A= np.triu(A, k=0)
       [row_index,col_index]=np.where(upper_A==1)

       ratio=0.1   # disturb how many links are removed
       select_num=int(len(row_index)*ratio)
       index=arange(0, (upper_A.sum()).sum())
       # print(index.shape)

       random.seed(0)
       random.shuffle(index)
       # np.random.shuffle(index)
       select_index=index[0:select_num]
       delta_A=np.zeros(shape=(num,num))
       for i in range(0,select_num):
          delta_A[row_index[select_index[i]]][col_index[select_index[i]]]=1
          delta_A[col_index[select_index[i]]][row_index[select_index[i]]]=1

       return delta_A,row_index,col_index,select_num

    def disturb_matrix_method(train_drug_drug_matrix):
       input_A=np.matrix(train_drug_drug_matrix)
       [num,num]=input_A.shape
       delta_A,row_index,col_index,select_num=MethodHub.generate_distrub_matrix(input_A)
       A=input_A-delta_A
       eigenvalues, eigenvectors = LA.eig(A)
       num_eigenvalues=len(eigenvalues)

       delta_eigenvalues=np.zeros(num_eigenvalues)
       for i in range(0,num_eigenvalues):
             delta_eigenvalues[i]=real((np.transpose(eigenvectors[:,i])*delta_A*eigenvectors[:,i])/(np.transpose(eigenvectors[:,i])*eigenvectors[:,i]))

       reconstructed_A=np.zeros(shape=(num,num))
       for i in range(0,num_eigenvalues):
           reconstructed_A=reconstructed_A+(eigenvalues[i]+delta_eigenvalues[i])*eigenvectors[:,i]*np.transpose(eigenvectors[:,i])

       reconstructed_A[np.where(input_A==1)]=1

       return_matrix=reconstructed_A+np.transpose(reconstructed_A)
       return return_matrix

def ensemble_method(drug_drug_matrix ,train_drug_drug_matrix,testPosision):
    chem_sim_similarity_matrix = np.matrix(np.loadtxt('2018/chem_sim.txt'))
    label_similarity_matrix = np.matrix(np.loadtxt('2018/label_mat.txt'))
    offlabel_similarity_matrix = np.matrix(np.loadtxt('2018/off_sim.txt'))

    # chem_sim_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    # target_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    #
    # transporter_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    # enzyme_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    #
    # pathway_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    # indication_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    #
    # label_similarity_matrix = np.asmatrix(np.loadtxt("sim_q.txt", 'float'))
    # offlabel_similarity_matrix =np.asmatrix(np.loadtxt("sim_q.txt", 'float'))

    multiple_matrix=[]

    print('********************************************************')
    predict_matrix=MethodHub.neighbor_method(chem_sim_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(label_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(offlabel_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)

    # print('*************************************************************************************************************************************')
    common_similarity_matrix,AA_similarity_matrix,RA_similarity_matrix,Katz_similarity_matrix,ACT_similarity_matrix,RWR_similarity_matrix=Topology.topology_similarity_matrix(train_drug_drug_matrix)
    predict_matrix=MethodHub.neighbor_method(common_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=common_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(AA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=AA_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(RA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RA_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(Katz_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=Katz_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(ACT_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=ACT_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.neighbor_method(RWR_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RWR_similarity_matrix

    multiple_matrix.append(predict_matrix)

    # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.Label_Propagation(chem_sim_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)  #12


    predict_matrix=MethodHub.Label_Propagation(label_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(offlabel_similarity_matrix,train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)   #14

    # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.Label_Propagation(common_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=common_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(AA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=AA_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(RA_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RA_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(Katz_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=Katz_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(ACT_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=ACT_similarity_matrix

    multiple_matrix.append(predict_matrix)

    predict_matrix=MethodHub.Label_Propagation(RWR_similarity_matrix,train_drug_drug_matrix)
    # predict_matrix=RWR_similarity_matrix

    multiple_matrix.append(predict_matrix)

   # print('*************************************************************************************************************************************')
    predict_matrix=MethodHub.disturb_matrix_method(train_drug_drug_matrix)

    multiple_matrix.append(predict_matrix)

    return multiple_matrix

def internal_determine_parameter(drug_drug_matrix):
    train_drug_drug_matrix,testPosition=holdout_by_link(copy.deepcopy(drug_drug_matrix),0.2,1)
    multiple_matrix =ensemble_method(copy.deepcopy(drug_drug_matrix),train_drug_drug_matrix,testPosition)
    weights=getParamter(copy.deepcopy(drug_drug_matrix),multiple_matrix,testPosition)
    # weights=[]

    input_matrix=[]
    output_matrix = []
    for i in range(0, len(testPosition)):
        vector=[]
        for j in range(0, len(multiple_matrix)):
           vector.append(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
        input_matrix.append(vector)
        output_matrix.append(drug_drug_matrix[testPosition[i][0], testPosition[i][1]])


    input_matrix=np.array(input_matrix)
    output_matrix= np.array(output_matrix)
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf1.fit(input_matrix, output_matrix)

    clf2 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    clf2.fit(input_matrix, output_matrix)
    print('*************************parameter determined*************************')
    # return weights
    return weights,clf1, clf2

def holdout_by_link(drug_drug_matrix, ratio, seed):
    link_number = 0
    link_position = []
    nonLinksPosition = []  # all non-link position
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])

    link_position = np.array(link_position)
    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)
    train_index = index[(int(link_number * ratio) + 1):]
    test_index = index[0:int(link_number * ratio)]
    train_index.sort()
    test_index.sort()
    testLinkPosition = link_position[test_index]
    train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)

    for i in range(0, len(testLinkPosition)):
        train_drug_drug_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
        train_drug_drug_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
    testPosition = list(testLinkPosition) + list(nonLinksPosition)

    return train_drug_drug_matrix, testPosition

def ensemble_scoringw(real_matrix, multiple_matrix, testPosition, weights ):
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability=np.array(predicted_probability)
        multiple_prediction.append(predicted_probability)
    ensemble_prediction = np.zeros(len(real_labels))
    for i in range(0, len(multiple_matrix)):
        ensemble_prediction = ensemble_prediction + weights[i] * multiple_prediction[i]
    normalize = MinMaxScaler()
    ensemble_prediction = normalize.fit_transform(ensemble_prediction)

    result = calculate_metric_scorew(real_labels, ensemble_prediction)

    return result
def calculate_metric_scorew(real_labels,predict_score):


   precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
   aupr_score = auc(recall, precision)

   all_F_measure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + precision[k]) > 0:
          all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          all_F_measure[k] = 0
   max_index = all_F_measure.argmax()
   threshold = pr_thresholds[max_index]
   fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
   auc_score = auc(fpr, tpr)

   predicted_score = np.zeros(len(real_labels))
   predicted_score[predict_score > threshold] = 1

   f = f1_score(real_labels, predicted_score)
   accuracy = accuracy_score(real_labels, predicted_score)
   precision = precision_score(real_labels, predicted_score)
   recall = recall_score(real_labels, predicted_score)
   print('results for feature:' + 'weighted_scoring')
   print(    '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f************************' % (
        auc_score, aupr_score, recall, precision, accuracy))

   auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)

   return auc_score,fpr, tpr, aupr_score, real_labels,predict_score

def ensemble_scoring1(real_matrix, multiple_matrix, testPosition, cf1 ):
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability=np.array(predicted_probability)
        multiple_prediction.append(predicted_probability)

    ensemble_prediction_cf1 = np.zeros(len(real_labels))

    for i in range(0, len(testPosition)):
        vector=[]
        for j in range(0, len(multiple_matrix)):
           vector.append(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
        vector=np.array(vector)

        aa=cf1.predict_proba(vector)

        ensemble_prediction_cf1[i]=(cf1.predict_proba(vector))[0][1]

    normalize = MinMaxScaler()



    result_cf1=calculate_metric_score1(real_labels, ensemble_prediction_cf1)

    return result_cf1
def calculate_metric_score1(real_labels,predict_score):


   precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
   aupr_score = auc(recall, precision)

   all_F_measure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + precision[k]) > 0:
          all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          all_F_measure[k] = 0
   max_index = all_F_measure.argmax()
   threshold = pr_thresholds[max_index]
   fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
   auc_score = auc(fpr, tpr)

   predicted_score = np.zeros(len(real_labels))
   predicted_score[predict_score > threshold] = 1

   f = f1_score(real_labels, predicted_score)
   accuracy = accuracy_score(real_labels, predicted_score)
   precision = precision_score(real_labels, predicted_score)
   recall = recall_score(real_labels, predicted_score)
   print('results for feature:' + 'c1_scoring')
   print(    '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f************************' % (
        auc_score, aupr_score, recall, precision, accuracy))

   auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)

   return auc_score,fpr, tpr, aupr_score, real_labels,predict_score

def ensemble_scoring2(real_matrix, multiple_matrix, testPosition,cf2):
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability=np.array(predicted_probability)
        multiple_prediction.append(predicted_probability)


    ensemble_prediction_cf2= np.zeros(len(real_labels))
    for i in range(0, len(testPosition)):
        vector=[]
        for j in range(0, len(multiple_matrix)):
           vector.append(multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
        vector=np.array(vector)

        ensemble_prediction_cf2[i]=(cf2.predict_proba(vector))[0][1]


    result_cf2=calculate_metric_score2(real_labels, ensemble_prediction_cf2)

    return  result_cf2
def calculate_metric_score2(real_labels,predict_score):


   precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
   aupr_score = auc(recall, precision)

   all_F_measure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + precision[k]) > 0:
          all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          all_F_measure[k] = 0
   max_index = all_F_measure.argmax()
   threshold = pr_thresholds[max_index]
   fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
   auc_score = auc(fpr, tpr)

   predicted_score = np.zeros(len(real_labels))
   predicted_score[predict_score > threshold] = 1

   f = f1_score(real_labels, predicted_score)
   accuracy = accuracy_score(real_labels, predicted_score)
   precision = precision_score(real_labels, predicted_score)
   recall = recall_score(real_labels, predicted_score)
   print('results for feature:' + 'weighted_scoring')
   print(    '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f************************' % (
        auc_score, aupr_score, recall, precision, accuracy))

   auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
   return auc_score,fpr, tpr, aupr_score, real_labels,predict_score

import timeit
a =timeit.default_timer()
runtimes = 1
drug_drug_matrix = np.loadtxt('2018/ddi_mat.txt',"int" )
# drug_drug_matrix = np.loadtxt("act_q.txt",dtype=int)
f, axes = pl.subplots(1, 2, figsize=(10, 5))

for seed in range(0, runtimes):

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='green', label=r'svm AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'SVM AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='green')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 2)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='green',linestyle='dashed', label=r'KNN AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'KNN AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='green',linestyle='dashed')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 3)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='black',linestyle='dashdot', label=r'LR AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'LR AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='black',linestyle='dashdot')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 4)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='blue',linestyle='dashdot', label=r'NB AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'NB AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='blue',linestyle='dashdot')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 5)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='y',linestyle='dashdot', label=r'DT AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'DT AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='y',linestyle='dashdot')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 0)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='yellow', label=r'vilar AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'vilar AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='yellow')


    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 6)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='blue', label=r'lP AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'lP AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='blue')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 7)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    axes[1].plot(mean_fpr, mean_tpr, color='cyan', label=r'Weighted AUC = %0.3f ' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Weighted AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='cyan')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='magenta', label=r'L1 AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'L1 AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='magenta')

    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 9)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr,  color='black',linestyle='dashed', label=r'L2 AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'L2 AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='black',linestyle='dashed')



    aucs, mean_fpr, tprs, y_real, y_proba = cross_validation(drug_drug_matrix, 5, seed, 10)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    axes[1].plot(mean_fpr, mean_tpr, color='red', label=r'GpsRf AUC = %0.3f' % (mean_auc), lw=2, alpha=.8)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'GpsRf AUpr=%.3f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='red')



    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('precision-recall curve ')
    axes[0].legend(loc='best', fontsize='small')

    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Roc curve ')
    axes[1].legend(loc="best", fontsize='small')
    pl.savefig('12018fknnsvmdar.png')
    pl.show()

b=timeit.default_timer()
c=(b- a)/60
print(c)

