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



headers1 = ['Walls U Value', 'Floor U Value',  'Roof U value',  'Maintenance Factor', 
                       'Av Frame Width','Glazing g', 'Frame U value', 'Psi Factor', 'Infiltration Rate', 'Ventilation Rate',
                      'Shelter Factor', 'Occupants', 'Metabolic Gains', 'Electrical Gains', 'MVHR efficiency', 'Window height', 'Window width',
                      'Alpha Angle', 'Beta Angle', 'Direction']            

w_height = []; w_width = []; w_alpha = []; w_beta = []; w_dirr = []
window_par_index = [21]
"""
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
    return sampleW, WindStDevP"""

def normalization(table):
    result = table.copy()
    for feature_name in table.columns:
        max_value = table[feature_name].max()
        min_value = table[feature_name].min()
        result[feature_name] = (table[feature_name] - min_value) / (max_value - min_value)
    return result

def analisys(datatable):
    
    """Pearson"""
    global resultsTable_reduced, pearsonListMean, betaListMean,rankBeta, L1ListMean, L2ListMean, WindMeanP, WindMeanW, WindStDevW, Table, corrMatrix, rankP, rankL1, rankL2 ,  pearson, beta, L1, L2
    
    resultsTable_reduced = datatable    
    corrAbs = corrMatrix = resultsTable_reduced.corr(method='pearson')
    corrAbs = corrMatrix.abs()  
      
    pearsonListMean = pearsonList = corrAbs['Total Heating E'].values.tolist() 
    
    #pearsonListMean = reduce_windows(pearsonList)[0]
    
  
    """Beta weight"""
    X = featuresTable = resultsTable_reduced.iloc[:, :-1]
    resultsList = resultsTable_reduced['Total Heating E'].tolist()     
    predictors = featuresTable.columns
  
    X = featuresTable[predictors]
    y = resultsTable_reduced['Total Heating E']
    
    # Initialise and fit model
    lm = LinearRegression()
    model = lm.fit(X, y)
    betaListMean = betaList = betas = model.coef_
   
    betaListMean = betaList = [abs(i) for i in betas] 
    #betaListMean = betaList = [i for i in betas]
    
    #betaListMean = reduce_windows(betaList)[0]  
    
    """LASSO"""
    clf = Lasso(alpha=0.05)
    clf.fit(featuresTable, resultsList)
    L1 = clf.coef_ #array
 
    L1ListMean = L1List = L1.tolist() 
    L1ListMean = L1List =  [abs(i) for i in L1List]    
    
    """Ridge"""
    clf = Ridge(alpha=1000)
    clf.fit(featuresTable, resultsList)
    L2 = clf.coef_ #array 
    L2List = L2.tolist() 
    L2ListMean = L2List =  [abs(i) for i in L2List]
    #L2ListMean = reduce_windows(L2List)[0]

    """All three"""   
    Table = pd.DataFrame(list(zip(pearsonListMean, betaListMean, L1ListMean, L2ListMean)), 
               columns =['pearson', 'beta', 'L1', 'L2'])

    rankP = Table["pearson"].rank(method = 'min') 
    rankBeta = Table["beta"].rank(method = 'min')
    rankL1 = Table["L1"].rank(method = 'min') 
    rankL2 = Table["L2"].rank(method = 'min') 

    Table['Rank_by_pearson'] = rankP #add new column with ranking
    Table['Rank_by_beta'] = rankBeta
    Table['Rank_by_L1'] = rankL1
    Table['Rank_by_L2'] = rankL2
    #Table = Normalisation(Table)
    #Table /= Table.max() 
    #Table = (Table-Table.min())/(Table.max()-Table.min())
    Table.to_csv('Table.csv', index=False)
 
    
"""Use one method based on various samples"""
samples = []
datasets = [sample2, sample4, ind_sample1, ind_sample3]
#datasets = [sample1, sample1, sample1, sample1]
method = "beta"

for i in datasets: #get colums with correlation method used on four different samples
    analisys(i) 
    samples.append(Table[method]) 
    
samplesDF = pd.DataFrame(list(zip(headers1, samples[0], samples[1], samples[2], samples[3])), 
               columns =['factor', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'], index = headers1)
 

"""Four methods based on the same sample """
#sample1, sample2, sample3, sample4 ind_sample1, ind_sample2, ind_sample3
test_sample = sample3.drop(['Walls Area', 'Floor Area', 'Roof Area', 'Mid-pane U value', 'TFA', 'Volume', 'Linear TB coef', 'length of TB','Point TB coef', 'Number of point TB', 'Floors', 'W2H', 'W2W', 'W2A', 'W2B', 'W2dir', 'W3H',
       'W3W', 'W3A', 'W3B', 'W3dir', 'W4H', 'W4W', 'W4A', 'W4B', 'W4dir'], axis=1)
analisys(test_sample)
Name = "realistic sample "

sb.set(color_codes=True)


RanksDF = pd.DataFrame(list(zip(headers1, pearsonListMean, betaListMean, L2ListMean, L1ListMean)), 
               columns =['factor', 'Pearson',  'Beta', 'Ridge','LASSO'], index = headers1)

#RanksDF = RanksDF.sort_values(by=['Ridge'], ascending=False)
RanksDF.to_csv('Arealistic.csv', index=True) 
#RanksDF.to_csv('Aindependent.csv', index=True) 


"""PAIRS """
grid = sb.pairplot(RanksDF)
plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight') 


#RanksDF2.plot.bar(rot=90, figsize=(25,6), color=['#9DE0AD', '#547980'], width=0.8, edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
       
ax = RanksDF.plot.bar(rot=90, figsize=(15,6), color=['#31a354', '#2c7fb8', '#594F4F', '#e62e00'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
#e60000
#6E97A3
plt.title(Name, loc='left')
plt.savefig('fig5.png',dpi=400, bbox_inches = 'tight') 



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



"""two samples on separate plots  """

sb.jointplot(x, y, data=samplesDF, kind="reg",color=('#F45941'), stat_func=r2)#real 
plt.savefig('fig2.png',dpi=400, bbox_inches = 'tight') 
sb.jointplot('Sample 3', 'Sample 4', data=samplesDF, kind="reg",color=('#468CB8'), stat_func=r2)#independant 
plt.savefig('fig3.png',dpi=400, bbox_inches = 'tight')  
# ci=None


"""Four samples on bar plot """
samplesDF.plot.bar(rot=90, figsize=(22,5), color=['#F45941', '#F45941', '#468CB8', '#468CB8'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.title(method, loc='left')
plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight') 


"""Averedge bar """
#Choose between rankP, rankBeta, rankL1, rankL2, betaListMean, pearsonListMean, L1ListMean, L2ListMean, 
allRanksList = [rankP, rankL1, rankL2]

AveregeRanksList = np.average(allRanksList, axis=0)
AveregeRanksDF = pd.DataFrame(list(zip(headers1, AveregeRanksList)), columns =['factor', 'Rank'], index = headers1)
AveregeRanksDF.plot.bar(rot=90, figsize=(15,6), color=['#F45941'], width=0.8,  edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
    
plt.title(Name, loc='left')
plt.savefig('fig6.png',dpi=400, bbox_inches = 'tight') 


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
plt.savefig('corrMethods.png',dpi=400, bbox_inches = 'tight') 
"""
      
print("--- %s seconds ---" % (time.time() - start_time)) 



 
        
        
        
        
        