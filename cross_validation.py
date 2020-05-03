import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time

start_time = time.time()

plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

start_time = time.time()
sample1 = pd.read_csv('realistic_data1.csv')
sample2 = pd.read_csv('realistic_data2.csv')
sample3 = pd.read_csv('realistic_data3.csv')
sample4 = pd.read_csv('realistic_data4.csv')
ind_sample1 = pd.read_csv('independent_data1.csv')
ind_sample2 = pd.read_csv('independent_data2.csv')
ind_sample3 = pd.read_csv('independent_data3.csv')



def Normalisation(table):
    names = table.columns
    scaler = preprocessing.StandardScaler()
    scaledTable = scaler.fit_transform(table)
    scaledTable = pd.DataFrame(scaledTable, columns=names)
    return scaledTable
sample1 = Normalisation(sample1)
sample2 = Normalisation(sample2)
sample3 = Normalisation(sample3)
sample4 = Normalisation(sample4)
ind_sample1 = Normalisation(ind_sample1)
ind_sample2 = Normalisation(ind_sample2)
ind_sample3 = Normalisation(ind_sample3)

headers1 = ['Walls U Value', 'Walls Area', 'Floor U Value', 'Floor Area', 'Roof U value', 'Roof Area', 'Maintenance Factor', 
                      'Mid-pane U value', 'Av Frame Width','Glazing g', 'Frame U value', 'Psi Factor', 'Infiltration Rate', 'Ventilation Rate',
                      'Shelter Factor', 'TFA', 'Volume', 'Occupants', 'Metabolic Gains', 'Electrical Gains', 'MVHR efficiency', 'Window height', 'Window width', 
                      'Alpha Angle', 'Beta Angle', 'Direction', 'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB']            

ind_sample2 = ind_sample2.drop(['Walls Area', 'Floor Area', 'Roof Area', 'Mid-pane U value', 'TFA', 'Volume', 'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB'], axis=1)
ind_sample3 = ind_sample3.drop(['Walls Area', 'Floor Area', 'Roof Area', 'Mid-pane U value', 'TFA', 'Volume', 'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB'], axis=1)


"""REALISTIC"""
sample = ind_sample3 #training 
featuresTrain = sample.iloc[:, :-1]
resultsTrain = sample['Total Heating E'].tolist()

sample = ind_sample2 #testing 
featuresTest = sample.iloc[:, :-1]
resultsTest = sample['Total Heating E'].tolist()

alpha = [0.0001, 0.05, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32,128, 256, 512, 1000]

resL1 = dict()
for a in alpha:
    Lasso1 = Lasso(alpha=a)
    resL1[a] = cross_val_score(Lasso1, featuresTrain, resultsTrain, scoring='neg_mean_squared_error', cv = 10)

resL1_df = pd.DataFrame(resL1)
choose_alpha = resL1_df.mean()
print("LASSO", choose_alpha)

Lasso2 = Lasso(alpha=0.05)
Lasso2.fit(featuresTrain, resultsTrain)
testpredict = Lasso2.predict(featuresTest)
err = mean_squared_error(resultsTest, testpredict)
print ("err Lasso real", err) #if higher that OLS then bad

L1Realistic = choose_alpha.tolist()
#L1Realistic = [round(num, 4) for num in L1Realistic] 

resL2 = dict()
for a in alpha:
    ridge1 = Ridge(alpha=a)
    resL2[a] = cross_val_score(ridge1, featuresTrain, resultsTrain, scoring='neg_mean_squared_error', cv = 10)
resL2_df = pd.DataFrame(resL2)
choose_alpha = resL2_df.mean()
L2Realistic = choose_alpha.tolist()
print("Ridge", choose_alpha)

ridge2 = Ridge(alpha=8)
ridge2.fit(featuresTrain, resultsTrain)
testpredict = ridge2.predict(featuresTest)
err = mean_squared_error(resultsTest, testpredict)
print ("err Ridge real", err) #if higher that OLS then bad


"""INDEPENDENT"""

sample = ind_sample3 #training 
featuresTrain = sample.iloc[:, :-1]
resultsTrain = sample['Total Heating E'].tolist()

sample = ind_sample2 #testing 
featuresTest = sample.iloc[:, :-1]
resultsTest = sample['Total Heating E'].tolist()

resL1 = dict()
for a in alpha:
    Lasso1 = Lasso(alpha=a)
    resL1[a] = cross_val_score(Lasso1, featuresTrain, resultsTrain, scoring='neg_mean_squared_error', cv = 10)
resL1_df = pd.DataFrame(resL1)
choose_alpha = resL1_df.mean()

L1indep = choose_alpha.tolist()
#L1Realistic = [round(num, 4) for num in L1Realistic]

Lasso2 = Lasso(alpha=0.05)
Lasso2.fit(featuresTrain, resultsTrain)
testpredict = Lasso2.predict(featuresTest)
err = mean_squared_error(resultsTest, testpredict)
print ("err Lasso indep", err) #if higher that OLS than bad



resL2 = dict()
for a in alpha:
    ridge1 = Ridge(alpha=a)
    resL2[a] = cross_val_score(ridge1, featuresTrain, resultsTrain, scoring='neg_mean_squared_error', cv = 10)
resL2_df = pd.DataFrame(resL2)
choose_alpha = resL2_df.mean()

L2indep = choose_alpha.tolist()

ridge2 = Ridge(alpha=256)
ridge2.fit(featuresTrain, resultsTrain)
testpredict = ridge2.predict(featuresTest)
err = mean_squared_error(resultsTest, testpredict)
print ("err Ridge indep", err)

df = pd.DataFrame(list(zip(alpha, L1Realistic, L2Realistic, L1indep, L2indep)), columns =['alpha', 'L1 realistic', 'L2 realistic','L1 independent','L2 independent'])

df.to_csv('Crossval.csv', index=False)