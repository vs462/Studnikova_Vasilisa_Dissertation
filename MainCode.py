import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time

from scipy.stats import pearsonr
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score



start_time = time.time()

plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

#Open files
sample1, sample2, sample3 = pd.read_csv('reducedData1.csv'), pd.read_csv('reducedData2.csv'), pd.read_csv('reducedData4.csv')
ind_sample1, ind_sample2, ind_sample3 = pd.read_csv('ind_reducedData.csv'), pd.read_csv('ind_reducedData2.csv'), pd.read_csv('ind_reducedData4.csv')

toDrop = ['Floors', 'W2H', 'W2W', 'W2A', 'W2B', 'W2dir', 'W3H', 'W3W', 'W3A', 
          'W3B', 'W3dir', 'W4H', 'W4W', 'W4A', 'W4B', 'W4dir']

#merge the available dataframes
realSample = pd.concat([sample1, sample2, sample3]).reset_index(drop=True).drop(toDrop, axis=1)
indSample = pd.concat([ind_sample1, ind_sample2, ind_sample3]).reset_index(drop=True).drop(toDrop,axis=1)

def Normalisation(table):
    table.rename(columns={'W1H': 'Window Height', 'W1W': 'Window Width', 'W1A': 'Alpha', 'W1B': 'Beta', 'W1dir': 'Direction'}, inplace=True)
    names = table.columns
    scaler = preprocessing.StandardScaler()
    #scaler = preprocessing.MinMaxScaler()
    scaledTable = scaler.fit_transform(table)
    scaledTable = pd.DataFrame(scaledTable, columns=names)
    return scaledTable

realSample = Normalisation(realSample)
indSample = Normalisation(indSample)

headers1 = features_names = realSample.columns


def analisys(df):
    global X, pearsonListMean, betaListMean,rankBeta, L1ListMean, L2ListMean, WindMeanP, WindStDevP, WindMeanW, WindStDevW, Table, corrMatrix, rankP, rankL1, rankL2 

    """Pearson"""
    corrAbs = df.corr(method='pearson').abs()
    pearsonListMean = corrAbs['Total Heating E'].values.tolist() 
    
    """ ML""" 
    X = df.iloc[:, :-1]
    y = df['Total Heating E']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    def evaluation(clf):
        print('R-squared score (training): {:.3f}'
         .format(clf.score(X_train, y_train)))
        print('R-squared score (test): {:.3f}'
         .format(clf.score(X_test, y_test)))
        
        kfold = model_selection.KFold(n_splits=10, random_state=0)
        
        scoring = 'neg_mean_squared_error'
        results = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring=scoring)
        print(("MAE: %.3f (%.3f)") % (results.mean(), results.std()))
        
        return results


    """OLS"""
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)
    betaListMean = [abs(i) for i in model.coef_]  
    
    #print('OLS')
    #accuracy = evaluation(lr)

    """LASSO"""
    linlasso = Lasso(alpha=0.05)
    linlasso.fit(X_train, y_train)
    L1ListMean =  [abs(i) for i in linlasso.coef_] 
    
    #print('linlasso')
    #accuracy = evaluation(linlasso)

    
    """Ridge"""
    linridge = Ridge(alpha=8)
    linridge.fit(X_train, y_train)
    L2ListMean =  [abs(i) for i in linridge.coef_]
    
    #print('linridge')
    #accuracy = evaluation(linridge)
    
    """Polynomial"""
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y,
                                                       random_state = 0)
    
    linreg = LinearRegression().fit(X_train_poly, y_train) 
    print('(poly deg 2) R-squared score (training): {:.3f}'
          .format(linreg.score(X_train_poly, y_train)))
    print('(poly deg 2) R-squared score (test): {:.3f}\n'
         .format(linreg.score(X_test_poly, y_test)))


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
    return X, pearsonListMean, betaListMean,rankBeta, L1ListMean, L2ListMean, Table, corrMatrix, rankP, rankL1, rankL2 
    
x = analisys(realSample)
 
    
"""Use one method based on various samples"""
samples = []
datasets = [sample2, sample3, ind_sample1, ind_sample3]
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
    return pearsonr(x, y)[0] ** 2
r21 =r2_score(samplesDF['Sample 1'], samplesDF['Sample 2'])
r22 =r2_score(samplesDF['Sample 3'], samplesDF['Sample 4'])
pearsonr1 = pearsonr(samplesDF['Sample 1'], samplesDF['Sample 2'])
pearsonr2 = pearsonr(samplesDF['Sample 3'], samplesDF['Sample 4'])
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

sb.jointplot('TFA', 'Total Heating E', data=df, kind="reg",color=('#F45941'), stat_func=r2)#real 
plt.savefig('fig2.png',dpi=400, bbox_inches = 'tight') 


sb.jointplot('Sample 3', 'Sample 4', data=samplesDF, kind="reg",color=('#468CB8'), stat_func=r2)#independant 
plt.savefig('fig3.png',dpi=400, bbox_inches = 'tight')  



"""Four samples on bar plot """
samplesDF.plot.bar(rot=90, figsize=(22,5), color=['#F45941', '#F45941', '#468CB8', '#468CB8'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.title(method, loc='left')
plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight') 



"""Pearson corr for methods"""
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

  
#Correlation Matrix
"""    
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
plt.savefig('corr.png',dpi=400, bbox_inches = 'tight') """


   
"""
methodsDF = pd.DataFrame(list(zip(headers1, rankP, rankL1, rankL2 )), 
               columns =['factor', 'pearson', 'L1', 'L2'], index = headers1)
methodsDF.plot.bar(rot=90, figsize=(40,10), color=['#418CBD', 'green', '#5D6D7E'], width=0.8, 
                    edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.savefig('result1.png',dpi=400, bbox_inches = 'tight')
"""


#two lists only with factors
"""pearsonH = PearsonSorted['factor'].tolist()

L1H = L1Sorted['factor'].tolist()

PearsonSorted.to_csv('ByPearson.csv',index=False)
L1Sorted.to_csv('ByL1.csv',index=False)"""


print("--- %s seconds ---" % (time.time() - start_time)) 