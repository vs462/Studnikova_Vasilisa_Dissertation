import pandas as pd
import matplotlib.pyplot as plt

headers1 = ['Walls U Value', 'Floor U Value',  'Roof U value',  'Maintenance Factor', 
                       'Av Frame Width','Glazing g', 'Frame U value', 'Psi Factor', 'Infiltration Rate', 'Ventilation Rate',
                      'Shelter Factor', 'Occupants', 'Metabolic Gains', 'Electrical Gains', 'MVHR efficiency', 'Window height', 'Window width',
                      'Alpha Angle', 'Beta Angle', 'Direction']            

results = pd.read_csv('all_methods_score.csv')

#head = results['factor'].tolist()
real = results['Realistic'].tolist()
indiv = results['Independent'].tolist()

results = pd.DataFrame(list(zip(real, indiv)), 
               columns =['Realistic', 'Independent'], index = headers1)

SortReal = results.sort_values(by=['Realistic'], ascending=True)
SortInd = results.sort_values(by=['Independent'], ascending=True)

SortReal.plot.bar(rot=90, figsize=(10,4), color=['#F45941', '#468CB8'], width=0.8, 
                  edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')

plt.savefig('fig4.png',dpi=400, bbox_inches = 'tight')


SortReal.plot.barh(rot=0, figsize=(4,10), color=['#F45941', '#468CB8'], width=0.8, 
                  edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')

plt.savefig('fig5.png',dpi=400, bbox_inches = 'tight')



results2 = pd.DataFrame(list(zip(indiv)), 
               columns =['Independent'], index = headers1)

SortReal = results.sort_values(by=['Realistic'], ascending=False)
SortInd = results2.sort_values(by=['Independent'], ascending=True)

SortInd.plot.barh(rot=0, figsize=(4,10), color=['#468CB8'], width=0.8, 
                  edgecolor='white').yaxis.grid(color='gray', linestyle='dashed')
plt.legend(loc='lower right')

plt.savefig('fig6.png',dpi=400, bbox_inches = 'tight')

