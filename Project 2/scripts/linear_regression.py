#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


from sklearn.cluster import KMeans
import numpy as np
import csv
import math


# Read the target vector csv file
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            t.append(int(row[0]))
    return t


# Read the input data vector csv file
def GenerateRawData(filePath):
    dataMatrix = []
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)

    dataMatrix = np.transpose(dataMatrix)
    return dataMatrix


# Make the Target values from the target dataset
def GenerateTrainingTarget(rawTraining,training_percent):
    TrainingLen = int(math.ceil(len(rawTraining)*(training_percent*0.01)))
    t           = rawTraining[:TrainingLen]
    return t


# Make the training data matrix from the raw dataset
def GenerateTrainingDataMatrix(rawData, training_percent):
    T_len = int(math.ceil(len(rawData[0])*0.01*training_percent))
    d2 = rawData[:,0:T_len]
    return d2


# GenerateValData() generates the data matrix for the Validation and Testing stages
def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    return dataMatrix


# GenerateValTargetVector() generates the target vector for the Validation and Testing stages
def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    return t


# GenerateBigSigma() generates the covariance matrix
def GenerateBigSigma(Data, MuMatrix,training_percent):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(training_percent*0.01))
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    # We only consider the diagonal values in the covriance matrix as we are only concerned with
    # the covariance of a feature with itself
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
        BigSigma = np.dot(200, BigSigma)

    return BigSigma


# The three functions GetScalar() , GetRadialBasisOut() and GetPhiMatrix together give us the
# equation for our Gaussian Radial Basis function for multiple variables

# This calculates our ( x - mu )
def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L


# The GetRadialBasisOut() forms the final equation of our equation by using math.exp()
# on values from GetScalar() and GetPhiMatrix()
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x


# The GetPhiMatrix() transposes the data from GetScalar()
def GetPhiMatrix(Data, MuMatrix, BigSigma, training_percent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(training_percent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    return PHI


# GetWeightsClosedForm() performs the Moore-Penrose Matix inversion to the PHI matrix to update the weights
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    return W


def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y


def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))

    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# Perform Linear Regression using Closed-Form Solution
def closed_form_solution(raw_data, raw_target, c_lambda, M, training_percent, validation_percent, test_percent):
    # Read the input and target datasets
    RawTarget = GetTargetVector(raw_target)
    RawData   = GenerateRawData(raw_data)

    # Generate the data matrix and target vectors for the Training phase
    TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,training_percent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,training_percent)

    # Generate the data matrix and target vectors for the Validatio phase
    ValDataAct = np.array(GenerateValTargetVector(RawTarget,validation_percent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,validation_percent, (len(TrainingTarget)))

    # Generate the data matrix and target vectors for the Testing phase
    TestDataAct = np.array(GenerateValTargetVector(RawTarget,test_percent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,test_percent, (len(TrainingTarget)+len(ValDataAct)))

    # Clustering using K-Means
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_

    BigSigma     = GenerateBigSigma(RawData, Mu, training_percent)
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, training_percent)
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(c_lambda))
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, test_percent)
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, validation_percent)

    TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
    VAL_TEST_OUT = GetValTest(VAL_PHI,W)
    TEST_OUT     = GetValTest(TEST_PHI,W)

    TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
    ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
    TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))

    return W, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct, TrainingAccuracy, ValidationAccuracy, TestAccuracy


# Perform Linear Regression using Gradient Descent
def gradient_descent_solution(W, c_lambda, learning_rate, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct):
    # Initializing the weights randomly for Gradient Descent
    W_Now        = np.dot(220, W)

    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []

    for i in range(0,400):

        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(c_lambda,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
        Delta_W       = -np.dot(learning_rate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next

        # TrainingData Accuracy
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        # ValidationData Accuracy
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        # TestingData Accuracy
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))

    return L_Erms_TR, L_Erms_Val, L_Erms_Test


def start():

    training_percent = 80
    validation_percent = 10
    test_percent = 10
    c_lambda = 2
    learning_rate = 0.01
    M = 25

    '''human_observed_features_dataset = read_from_CSV(r'../datasets/HumanObserved-Features-Data/HumanObserved-Features-Data.csv')
    gsc_features_dataset = read_from_CSV(r'../datasets/GSC-Features-Data/GSC-Features.csv')

    human_observed_same_pairs = read_from_CSV(r'../datasets/HumanObserved-Features-Data/same_pairs.csv')
    human_observed_diffn_pairs = read_from_CSV(r'../datasets/HumanObserved-Features-Data/diffn_pairs.csv')

    gsc_same_pairs = read_from_CSV(r'../datasets/GSC-Features-Data/same_pairs.csv')
    gsc_diffn_pairs = read_from_CSV(r'../datasets/GSC-Features-Data/diffn_pairs.csv')

    human_observed_same_pairs_concat_dataset = generate_concat_dataset(human_observed_features_dataset, human_observed_same_pairs, 2, 750)
    human_observed_diffn_pairs_concat_dataset = generate_concat_dataset(human_observed_features_dataset, human_observed_diffn_pairs, 2, 750)

    human_observed_same_pairs_difference_dataset = generate_difference_dataset(human_observed_features_dataset, human_observed_same_pairs, 2, 750)
    human_observed_diffn_pairs_difference_dataset = generate_difference_dataset(human_observed_features_dataset, human_observed_diffn_pairs, 2, 750)

    gsc_same_pairs_concat_dataset = generate_concat_dataset(gsc_features_dataset, gsc_same_pairs, 1, 1000)
    gsc_diffn_pairs_concat_dataset = generate_concat_dataset(gsc_features_dataset, gsc_diffn_pairs, 1, 1000)

    gsc_same_pairs_difference_dataset = generate_difference_dataset(gsc_features_dataset, gsc_same_pairs, 1, 1000)
    gsc_diffn_pairs_difference_dataset = generate_difference_dataset(gsc_features_dataset, gsc_diffn_pairs, 1, 1000)

    human_observed_concat_dataset = merge_dataset(human_observed_same_pairs_concat_dataset, human_observed_diffn_pairs_concat_dataset)
    human_observed_concat_dataset = human_observed_concat_dataset.sample(frac = 1)

    human_observed_difference_dataset = merge_dataset(human_observed_same_pairs_difference_dataset, human_observed_diffn_pairs_difference_dataset)
    human_observed_difference_dataset = human_observed_difference_dataset.sample(frac = 1)

    gsc_cat_dataset = merge_dataset(gsc_same_pairs_concat_dataset, gsc_diffn_pairs_concat_dataset)
    gsc_cat_dataset = gsc_cat_dataset.sample(frac = 1) # Shuffling the dataset
    gsc_cattest_dataset = gsc_cat_dataset.iloc[:int(math.ceil(gsc_cat_dataset.shape[0]*training_percent*0.01)), : ]

    # Dropping features that give same value for every sample (Features that do not contribute to our model)
    cols = list(gsc_cattest_dataset)
    nunique = gsc_cattest_dataset.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    gsc_concat_dataset = gsc_cat_dataset.drop(cols_to_drop, axis = 1)

    gsc_diff_dataset = merge_dataset(gsc_same_pairs_difference_dataset, gsc_diffn_pairs_difference_dataset)
    gsc_diff_dataset = gsc_diff_dataset.sample(frac = 1) # Shuffling the dataset
    gsc_difftest_dataset = gsc_diff_dataset.iloc[:int(math.ceil(gsc_diff_dataset.shape[0]*training_percent*0.01)), : ]

    # Dropping features that give same value for every sample (Features that do not contribute to our model)
    cols = list(gsc_difftest_dataset)
    nunique = gsc_difftest_dataset.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    gsc_difference_dataset = gsc_diff_dataset.drop(cols_to_drop, axis = 1)

    human_observed_concat_raw_data = generate_data(human_observed_concat_dataset)
    human_observed_concat_raw_target = generate_target(human_observed_concat_dataset)

    human_observed_difference_raw_data = generate_data(human_observed_difference_dataset)
    human_observed_difference_raw_target = generate_target(human_observed_difference_dataset)

    gsc_concat_raw_data = generate_data(gsc_concat_dataset)
    gsc_concat_raw_target = generate_target(gsc_concat_dataset)

    gsc_difference_raw_data = generate_data(gsc_difference_dataset)
    gsc_difference_raw_target = generate_target(gsc_difference_dataset)

    write_to_CSV(human_observed_concat_raw_data, r'../datasets/human_observed_concat_raw_data.csv')
    write_to_CSV(human_observed_concat_raw_target, r'../datasets/human_observed_concat_raw_target.csv')

    write_to_CSV(human_observed_difference_raw_data, r'../datasets/human_observed_difference_raw_data.csv')
    write_to_CSV(human_observed_difference_raw_target, r'../datasets/human_observed_difference_raw_target.csv')

    write_to_CSV(gsc_concat_raw_data, r'../datasets/gsc_concat_raw_data.csv')
    write_to_CSV(gsc_concat_raw_target, r'../datasets/gsc_concat_raw_target.csv')

    write_to_CSV(gsc_difference_raw_data, r'../datasets/gsc_difference_raw_data.csv')
    write_to_CSV(gsc_difference_raw_target, r'../datasets/gsc_difference_raw_target.csv')'''


    print ("M = " + str(M) + "\nLambda  = " + str(c_lambda) + "\nLearning Rate = " + str(learning_rate))

    # Linear Regression on Human Observed Concatenated Features
    W, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct, TrainingAccuracy, ValidationAccuracy, TestAccuracy = closed_form_solution(r'../datasets/human_observed_concat_raw_data.csv', r'../datasets/human_observed_concat_raw_target.csv', c_lambda, M, training_percent, validation_percent, test_percent)
    L_Erms_TR, L_Erms_Val, L_Erms_Test = gradient_descent_solution(W, c_lambda, learning_rate, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct)
    print ('-------Linear Regression on Human Observed Concatenated Features-------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ()

    # Linear Regression on Human Observed Subtracted Features
    W, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct, TrainingAccuracy, ValidationAccuracy, TestAccuracy = closed_form_solution(r'../datasets/human_observed_difference_raw_data.csv', r'../datasets/human_observed_difference_raw_target.csv', c_lambda, M, training_percent, validation_percent, test_percent)
    L_Erms_TR, L_Erms_Val, L_Erms_Test = gradient_descent_solution(W, c_lambda, learning_rate, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct)
    print ('-------Linear Regression on Human Observed Differentiated Features-------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ()

    # Linear Regression on GSC Concatenated Features
    W, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct, TrainingAccuracy, ValidationAccuracy, TestAccuracy = closed_form_solution(r'../datasets/gsc_concat_raw_data.csv', r'../datasets/gsc_concat_raw_target.csv', c_lambda, M, training_percent, validation_percent, test_percent)
    L_Erms_TR, L_Erms_Val, L_Erms_Test = gradient_descent_solution(W, c_lambda, learning_rate, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct)
    print ('-------Linear Regression on GSC Concatenated Features-------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ()

    # Linear Regression on GSC Subtracted Features
    W, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct, TrainingAccuracy, ValidationAccuracy, TestAccuracy = closed_form_solution(r'../datasets/gsc_difference_raw_data.csv', r'../datasets/gsc_difference_raw_target.csv', c_lambda, M, training_percent, validation_percent, test_percent)
    L_Erms_TR, L_Erms_Val, L_Erms_Test = gradient_descent_solution(W, c_lambda, learning_rate, TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingTarget, ValDataAct, TestDataAct)
    print ('-------Linear Regression on GSC Differentiated Features-------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ()


if __name__ == '__main__':
    start()
