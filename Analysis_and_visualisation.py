import numpy as np
import pandas as pd
import seaborn as sb
import scipy
from scipy import stats
from statistics import mean
from sklearn import preprocessing
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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


w_height = []; w_width = []; w_alpha = []; w_beta = []; w_dirr = []
window_par_index = [21]

def reduce_windows(sample):  #get averages for windows parameters 
    global sampleW, WBmeans
    for i in window_par_index:
        w_height.append(sample[i])
        w_width.append(sample[i+1])
        w_alpha.append(sample[i+2])
        w_beta.append(sample[i+3])
        w_dirr.append(sample[i+4])         
    #Wmeans = [np.average(w_height), np.average(w_width), np.average(w_alpha), np.average(w_beta), np.average(w_dirr)]
    #Wmeans = [w_height, w_width, w_alpha, w_beta, w_dirr]  
    sampleW = sample[:25]
    sampleW.extend(sample[41:]) 
    WindStDevP = 1    
    return sampleW, WindStDevP


def analisys(datatable):
    
    """Pearson"""
    global X, pearsonListMean, betaListMean,rankBeta, L1ListMean, L2ListMean, WindMeanP, WindStDevP, WindMeanW, WindStDevW, Table, corrMatrix, rankP, rankL1, rankL2 
    
    resultsTable_reduced = datatable  
    corrAbs = corrMatrix = resultsTable_reduced.corr(method='pearson')
    corrAbs = corrMatrix.abs()    
    
    pearsonListMean = pearsonList = corrAbs['Total Heating E'].values.tolist() 
    pearsonListMean = reduce_windows(pearsonList)[0]
    
  
    """Beta weight"""
    X = featuresTable = resultsTable_reduced.iloc[:, :-1]
    resultsList = resultsTable_reduced['Total Heating E'].tolist()     
    predictors = featuresTable.columns
  
    X = featuresTable[predictors]
    y = resultsTable_reduced['Total Heating E']
    
    # Initialise and fit model
    lm = LinearRegression()
    model = lm.fit(X, y)
    betas = model.coef_
    betaListMean = betaList = [abs(i) for i in betas] 
    betaListMean = reduce_windows(betaList)[0]  
    
    
    """LASSO"""
    clf = Lasso(alpha=0.05)
    clf.fit(featuresTable, resultsList)
    L1 = clf.coef_ #array
    L1ListMean = L1List = L1.tolist() 
    L1ListMean = L1List =  [abs(i) for i in L1List]    
    #L1List = np.array(L1List)
    #normalized_X = preprocessing.normalize([L1List])
    #L1List = L1List.tolist() 

    L1ListMean = reduce_windows(L1List)[0]  
    
    #correlation matrix
    #clf.fit(featuresTable, featuresTable)
    #corrMatrix2 = clf.coef_
    
    """Ridge"""
    clf = Ridge(alpha=8)
    clf.fit(featuresTable, resultsList)
    L2 = clf.coef_ #array 
    L2List = L2.tolist() 
    L2List =  [abs(i) for i in L2List]
    L2ListMean = reduce_windows(L2List)[0]

    """All three"""   
    Table = pd.DataFrame(list(zip(headers1, pearsonListMean, betaListMean, L1ListMean, L2ListMean)), 
               columns =['factor', 'pearson', 'beta', 'L1', 'L2'])

    rankP = Table["pearson"].rank(method = 'min') 
    rankBeta = Table["beta"].rank(method = 'min')
    rankL1 = Table["L1"].rank(method = 'min') 
    rankL2 = Table["L2"].rank(method = 'min') 

    Table['Rank_by_pearson'] = rankP #add new column with ranking
    Table['Rank_by_beta'] = rankBeta
    Table['Rank_by_L1'] = rankL1
    Table['Rank_by_L2'] = rankL2
    #PearsonSorted = Table.sort_values(by=['pearson'], ascending=False)
    #L1Sorted = Table.sort_values(by=['L1'], ascending=False)
    Table.to_csv('Table.csv', index=False)
 
    
"""Use one method based on various samples"""
samples = []
datasets = [sample2, sample4, ind_sample1, ind_sample3]
#datasets = [sample1, sample1, sample1, sample1]

method = "L2"
#sample1, sample2, ind_sample1, ind_sample2

for i in datasets: #get colums with correlation method used on four different samples
    analisys(i) 
    samples.append(Table[method]) 
    
samplesDF = pd.DataFrame(list(zip(headers1, samples[0], samples[1], samples[2], samples[3])), 
               columns =['factor', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'], index = headers1)


"""Four methods based on the same sample """
#sample1, sample2, sample3, sample4 ind_sample1, ind_sample2, ind_sample3
test_sample = sample3
analisys(test_sample)
Name = "realistic sample "

#Choose between rankP, rankBeta, rankL1, rankL2, betaListMean, pearsonListMean, L1ListMean, L2ListMean, 
sb.set(color_codes=True)

RanksDF = pd.DataFrame(list(zip(headers1, rankP, rankBeta, rankL2, rankL1)), 
               columns =['factor', 'Pearson', 'Beta', 'Ridge', 'LASSO'], index = headers1)
RanksDF2 = pd.DataFrame(list(zip(headers1, rankP, rankL2)), 
               columns =['factor', 'Pearson', 'Ridge'], index = headers1) 

RanksDF = pd.DataFrame(list(zip(headers1, pearsonListMean, betaListMean, L2ListMean, L1ListMean)), 
               columns =['factor', 'Pearson', 'Beta', 'Ridge', 'LASSO'], index = headers1)



"""PAIRS """
grid = sb.pairplot(RanksDF)
plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight') 

#RanksDF2.plot.bar(rot=90, figsize=(25,6), color=['#9DE0AD', '#547980'], width=0.8, edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
       
ax = RanksDF.plot.bar(rot=90, figsize=(18,4), color=['#31a354', '#2c7fb8', '#594F4F', '#6E97A3'], width=0.9, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.title(Name, loc='left')
plt.savefig('fig5.png',dpi=400, bbox_inches = 'tight') 

""" vertical
ax2 = RanksDF.plot.barh(rot=0, figsize=(4,18), color=['#31a354', '#2c7fb8', '#594F4F'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.title(Name, loc='left')
plt.savefig('fig6.png',dpi=400, bbox_inches = 'tight')  """




"""Two reg lines on the same plot """
x = "Sample 1"; y = "Sample 2"

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
r21 =r2_score(samplesDF['Sample 1'], samplesDF['Sample 2'])
r22 =r2_score(samplesDF['Sample 3'], samplesDF['Sample 4'])
pearsonr1 = scipy.stats.pearsonr(samplesDF['Sample 1'], samplesDF['Sample 2'])
pearsonr2 = scipy.stats.pearsonr(samplesDF['Sample 3'], samplesDF['Sample 4'])
print("R2 realistic = ", r21, "Pear realistic = ", pearsonr1)
print("R2  indep = ", r22, "Pear indep = ", pearsonr2)       

fig, ax = plt.subplots()
fig.set_size_inches(5, 5)
ax = sb.regplot(samplesDF["Sample 1"],samplesDF["Sample 2"],color=('#F45941'), scatter_kws={"s": 10}, truncate = True, label = 'Realistic')
ax2 = sb.regplot(samplesDF['Sample 3'],samplesDF['Sample 4'],color=('#468CB8'), scatter_kws={"s": 10}, truncate = True, label = 'Independent')                                                          
#plt.xlim(0, 31); plt.ylim(0, 31)# set the ylim to bottom, top  
ax.legend(loc="upper left", markerscale = 2)
ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
plt.title(method, loc='left')
plt.savefig('fig1.png',dpi=400, bbox_inches = 'tight')           



"""two samples on separate plots """

sb.jointplot('TFA', 'Total Heating E', data=test_sample, kind="reg",color=('#F45941'), stat_func=r2)#real 
plt.savefig('fig2.png',dpi=400, bbox_inches = 'tight') 




sb.jointplot('Sample 3', 'Sample 4', data=samplesDF, kind="reg",color=('#468CB8'), stat_func=r2)#independant 
plt.savefig('fig3.png',dpi=400, bbox_inches = 'tight')  
#, ci=None


"""Four samples on bar plot """
samplesDF.plot.bar(rot=90, figsize=(22,5), color=['#F45941', '#F45941', '#468CB8', '#468CB8'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.title(method, loc='left')
plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight') 



"""Pearson corr for methods
corrMatrix = RanksDF.corr(method='pearson')
pearson = sb.heatmap(corrMatrix, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns,
        #cmap='RdBu_r',
        cmap='Blues',
        annot=True,
        linewidth=0.5)
plt.title(Name, loc='left')
bottom, top = pearson.get_ylim()
pearson.set_ylim(bottom + 0.5, top - 0.5)   
plt.savefig('corrMethods.png',dpi=400, bbox_inches = 'tight')"""



#Correlation Matrix
    
clt = Ridge(alpha=10)
model = clt.fit(sample3, sample3)

betas = model.coef_
#alpha = model.intercept_
betaList = [abs(i) for i in betas] 
betaListMean = reduce_windows(betaList)[0]  

corrMatrix2 = betaList

plt.subplots(figsize=(15,12)) 
pearson = sb.heatmap(corrMatrix2, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns,
        #cmap='RdBu_r',
        cmap='Blues',
        #annot=True,
        linewidth=0.5)
bottom, top = pearson.get_ylim()
pearson.set_ylim(bottom + 0.5, top - 0.5)   
plt.savefig('corr.png',dpi=400, bbox_inches = 'tight') 

   
methodsDF = pd.DataFrame(list(zip(headers1, rankP, rankL1, rankL2 )), 
               columns =['factor', 'pearson', 'L1', 'L2'], index = headers1)
methodsDF.plot.bar(rot=90, figsize=(40,10), color=['#418CBD', 'green', '#5D6D7E'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.savefig('result1.png',dpi=400, bbox_inches = 'tight')


"""
def best_fit_line(x,y):
    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)*mean(x)) - mean(x*x)))
    
    b = mean(y) - m*mean(x)
    
    return m, b
x = samplesDF['sample1']
y = samplesDF['sample2']

m, b = best_fit_line(x,y)

regression_line = [(m*x)+b for x in x]

plt.scatter(x,y,color='#003F72')
plt.plot(x, regression_line)
plt.show()"""

      
print("--- %s seconds ---" % (time.time() - start_time)) 
   
        
        