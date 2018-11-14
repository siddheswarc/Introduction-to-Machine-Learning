from sklearn.cluster import KMeans
import numpy as np
import csv
import math


C_Lambda = 0.15 # Regularization term for Closed form solution
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10 # Number of basis functions
PHI = []


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


    # We discard five features out of the forty-six features as they do not contribute to our model
    dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)

    dataMatrix = np.transpose(dataMatrix)
    return dataMatrix


# Make the Target values from the target dataset
def GenerateTrainingTarget(rawTraining,TrainingPercent):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    return t


# Make the training data matrix from the raw dataset
def GenerateTrainingDataMatrix(rawData, TrainingPercent):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
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
def GenerateBigSigma(Data, MuMatrix,TrainingPercent):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
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
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
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
    #t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))

    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


## Fetch and Prepare Dataset

# Read the input and target datasets
RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv')
#print(type(RawData))


## Prepare Training Data

# Generate the data matrix and target vectors for the Training phase
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)


## Prepare Validation Data

# Generate the data matrix and target vectors for the Validatio phase
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))


## Prepare Test Data

# Generate the data matrix and target vectors for the Testing phase
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))


## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# Clustering using K-Means
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, TestPercent)
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, ValidationPercent)

## Finding Erms on training, validation and test set

TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


print ("-------Closed Form with Radial Basis Function-------")
print ("M = " + str(M) + "\nLambda = " + str(C_Lambda))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ()


## Gradient Descent solution for Linear Regression

# Initializing the weights randomly for Gradient Descent
W_Now        = np.dot(220, W)

# Regularization term for Gradient Descent
La           = 0.15

learningRate = 0.15
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []

for i in range(0,400):

    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next

    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))

    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))

    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


print ('----------Gradient Descent Solution--------------------')
print ("M = " + str(M) + "\nLambda  = " + str(La) + "\nLearning Rate = " + str(learningRate))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
print ()
