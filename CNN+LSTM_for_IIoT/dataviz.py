import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import os

sns.set_style("whitegrid")

def add_median_labels(ax, fmt='.3f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='normal', color='white',fontsize='smaller')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def PrepareData(dataframe, model):
    res = pd.DataFrame()
    for m in metrics:
        met = pd.DataFrame()
        met['Values'] = dataframe[m]
        met['Model']=model
        met['Metric']=m
        met['Mode'] = dataframe['Mode']
        res =pd.concat([met,res], ignore_index=True)
    return res

metrics = ['accuracy','precision','recall','f1-score' ]
columns = ['name','Mode']+metrics
#df.loc[df['name']=='Dataset_Authors'].boxplot(column=['accuracy','precision','recall','f1-score' ],by=['name'],grid=True, layout=(1,4))

dfb = pd.read_csv('result_report.Label-Final.csv')
dfb['Mode']='Binary'
dfm = pd.read_csv('result_report.Type-Final.csv')
dfm['Mode']='Multiclass'

df = pd.concat([dfb,dfm], ignore_index=True)
adt = df.loc[df['name']=='Dataset_Authors'][columns]#['name','accuracy','precision', 'recall','f1-score']]
pdt = df.loc[df['name']=='Proposed_Hybrid_CNN_LSTM'][columns]#['name','accuracy','precision', 'recall','f1-score']]
idt = df.loc[df['name']=='Inspired_CNN_LSTM'][columns]#[['name','accuracy','precision', 'recall','f1-score']]

#adt =adt.loc[adt['Mode']=='Binary'] # romove multiclass to not prejudice the visualization
print('subset metrics plots')
dfm.loc[dfm.name=='Proposed_Hybrid_CNN_LSTM'].plot(x='set',y=['accuracy','precision','recall','f1-score'],subplots=False,rot=0).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
dfb.loc[dfb.name=='Proposed_Hybrid_CNN_LSTM'].plot(x='set',y=['accuracy','precision','recall','f1-score'],subplots=False,rot=0).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

print('************************************************')
print('Metrics for Dataset Authors Binary Classification:')
print(adt[adt['Mode']=='Binary'].describe()[1:2])
print('************************************************')
print('************************************************')
print('Metrics for Dataset Authors Multiclass Classification:')
print(adt[adt['Mode']=='Multiclass'].describe()[1:2])
print('************************************************')

print('Show multiclass and binary for DatasetAuthors:')
g = sns.boxplot(y="Metric", x="Values",
                hue="Mode",
                data=PrepareData(adt,'Dataset'),
                orient='h',palette='pastel', linewidth=1, showfliers = False, showmeans=False)
add_median_labels(g,fmt='.3f')
plt.show()
plt.clf()

istat=pd.DataFrame([idt[idt['Mode']=='Multiclass'].std(),
                idt[idt['Mode']=='Multiclass'].median(),
                idt[idt['Mode']=='Multiclass'].mean(),
                idt[idt['Mode']=='Binary'].std(),
                idt[idt['Mode']=='Binary'].median(),
                idt[idt['Mode']=='Binary'].mean()
]
                 ,index=['Multiclass','Multiclass','Multiclass','Binary','Binary','Binary'])
istat['fun']=list(['Std','Median','Mean','Std','Median','Mean'])

pstat=pd.DataFrame([pdt[pdt['Mode']=='Multiclass'].std(),
                pdt[pdt['Mode']=='Multiclass'].median(),
                pdt[pdt['Mode']=='Multiclass'].mean(),
                pdt[pdt['Mode']=='Binary'].std(),
                pdt[pdt['Mode']=='Binary'].median(),
                pdt[pdt['Mode']=='Binary'].mean()
]
                 ,index=['Multiclass','Multiclass','Multiclass','Binary','Binary','Binary'])
pstat['fun']=list(['Std','Median','Mean','Std','Median','Mean'])

AMC = pd.DataFrame()
AMC['Std']=adt[adt['Mode']=='Multiclass'].std()
AMC['Median']=adt[adt['Mode']=='Multiclass'].median()
AMC['Mean']=adt[adt['Mode']=='Multiclass'].mean()
AMC['Mode']= 'Multiclass'
ABIN = pd.DataFrame()
ABIN['Std']=adt[adt['Mode']=='Binary'].std()
ABIN['Median']=adt[adt['Mode']=='Binary'].median()
ABIN['Mean']=adt[adt['Mode']=='Binary'].mean()
ABIN['Mode']= 'Binary'
resuadt = pd.concat([ABIN,AMC])
resuadt['Model']='Dataset Authors'

IMC = pd.DataFrame()
IMC['Std']=idt[idt['Mode']=='Multiclass'].std()
IMC['Median']=idt[idt['Mode']=='Multiclass'].median()
IMC['Mean']=idt[idt['Mode']=='Multiclass'].mean()
IMC['Mode']= 'Multiclass'
IBIN = pd.DataFrame()
IBIN['Std']=idt[idt['Mode']=='Binary'].std()
IBIN['Median']=idt[idt['Mode']=='Binary'].median()
IBIN['Mean']=idt[idt['Mode']=='Binary'].mean()
IBIN['Mode']= 'Binary'
resuidt = pd.concat([IBIN,IMC])
resuidt['Model']='Inspired'

summary = pd.concat([resuadt,resuidt])
summary.to_csv('resut_summary.csv')




print('Show multiclass and binary for Proposed:')
l = sns.boxplot(y="Metric", x="Values",
                hue="Mode",
                data=PrepareData(pdt,'Proposed'),
                orient='h',palette='pastel', linewidth=1, showfliers = False, showmeans=True)
#add_median_labels(l,fmt='.3f')
plt.show()
plt.clf()


print('Show multiclass and binary for Inspired:')
l = sns.boxplot(y="Metric", x="Values",
                hue="Mode",
                data=PrepareData(idt,'Inspired'),
                orient='h',palette='pastel', linewidth=1, showfliers = False, showmeans=True)
add_median_labels(l,fmt='.3f')
plt.show()
plt.clf()


x=PrepareData(idt,'Inspired')



data= pd.concat([PrepareData(pdt,'Proposed'),PrepareData(adt,'Dataset'),PrepareData(idt,'Inspired')], ignore_index=True)
#data= pd.concat([PrepareData(adt,'Dataset'),PrepareData(idt,'Inspired')], ignore_index=True)
s =sns.boxplot(data=data, x='Values',y='Metric',hue='Model', palette='pastel', linewidth=1, orient='h', showmeans=True)
add_median_labels(s,fmt='.3f')
plt.show()
plt.clf()


g = sns.catplot(y="Metric", x="Values",
                hue="Model", row='Mode',
                data=data, kind="box",
                orient='h',palette='pastel', linewidth=1,legend_out=False, showfliers = False)




g = sns.boxplot(y="Metric", x="Values",
                hue="Mode",
                data=PrepareData(adt,'Dataset'),
                orient='h',palette='pastel', linewidth=1, showfliers = False)


g = sns.catplot(y="Metric", x="Values",
                hue="Model", row='Mode',
                data=data, kind="box",
                height=4, aspect=.7, orient='h')
'''    
        met = pd.DataFrame()
        met['Values'] = idt[m]
        met['Model']='Inspired'
        met['Metric']=m
        res =pd.concat([met,res], ignore_index=True)
        '''



'''
s =sns.boxplot(data=res, x='metric',y='values')

adt.boxplot(column=metrics,by=['name'],grid=True, layout=(1,4))
plt.show()
'''