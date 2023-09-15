# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(PV)s
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

def Gompertz2(x,a,k,t): 
    return a*np.exp(-np.exp(-k*(x-t)))



pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.max_rows = None

df=pd.read_excel('bodyweightclean4.xlsx')

df_grace_male=pd.read_excel('grace_male.xlsx')
df_grace_female=pd.read_excel('grace_female.xlsx')
#print (df.head())
# color blind friendly palette from https://gist.github.com/thriveth/8560036
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
palette_dict ={'Sprague Dawley': '#ff7f00', 'Wistar':'#4daf4a' ,'ND':'#f781bf'}

df['Strain_sex']=df.Strain +' '+df.Sex
df['Strain_sex_auth']=df.Strain_sex +' '+df.Reference

#%% fig 1
fign=0

figure=plt.figure(fign)

gs = GridSpec(nrows=2, ncols=2)
ax0 = figure.add_subplot(gs[0, :])

sns.lineplot(data=df, x='DAF', y='Value', hue= 'Reference',style="Sex", legend=False,alpha= 0.5) #style="string_sex"

plt.ylabel("Body Weight (g)")
plt.title("Age vs Body Weight in rats")

sns.scatterplot( data=df,x='DAF', y='Value', hue= 'Reference',style="Sex",alpha= 0.5)
plt.legend(loc="lower right",ncol=3)
plt.savefig("fig/"+str(fign) +'a'+'.png')


ax1 = figure.add_subplot(gs[1, 0])
sns.lineplot(data=df, x='DAF', y='Value', hue= 'Reference',style="Sex", legend=False,alpha= 0.5) #style="string_sex"

plt.ylabel("Body Weight (g)")
#plt.title("Age vs Body Weight in rats")
plt.xlim(0,40)
plt.ylim(0,15)
sns.scatterplot( data=df,x='DAF', y='Value', hue= 'Reference',style="Sex",alpha= 0.5,legend=False)
ax2 = figure.add_subplot(gs[1, 1])
plt.axis('off')

plt.tight_layout()
fign=fign+1
plt.savefig("fig/"+str(fign) +'b'+'.png')
#%% fig  3 and sup 1

dfiur= df.loc[df['Reference'].isin(['Norman 1979','Witlin 2002','Schneidereit 1985','Zhang 2010'])]


dfeur=df[-df['Reference'].isin(['Norman 1979','Witlin 2002','Schneidereit 1985','Zhang 2010'])]

figure=plt.figure(fign)
fign=fign+1
sigma = np.ones(len(dfiur.DAF))
sigma[[0]] = 0.018
xnew_diur = np.arange(dfiur.DAF.min(), dfiur.DAF.max())
parameters_diur, pcov_diur = curve_fit(Gompertz2, dfiur.DAF, dfiur.Value,p0=[ 380, 0.035,73], sigma=sigma)
print (dfiur)
y_gomp_diur = Gompertz2(  xnew_diur, parameters_diur[0], parameters_diur[1], parameters_diur[2] )

sns.scatterplot(data=dfiur,x=dfiur.DAF,y= dfiur.Value,hue= 'Reference',style='Strain_sex')
plt.ylabel('Weitght (g)')

gb1 = dfeur.groupby(['Strain_sex_auth'])
plt.savefig("fig/"+str(fign) +'.png')
# for name, group in gb1:### logistic fit
    
#     group=pd.concat([dfiur,group])
#     xnew = np.arange(group.DAF.min(), group.DAF.max())
#     parameters, pcov = curve_fit(c_logistic, group.DAF, group.Value)
#     yaj = c_logistic(  xnew, parameters[0], parameters[1], parameters[2], )#parameters[3]
#     plt.scatter(group.DAF, group.Value, label=group.Strain_sex_auth.iloc[0])
#     plt.legend(loc="lower right")
#     plt.plot(xnew,yaj)
#     plt.show()

dfparams=pd.DataFrame(columns=dfeur.columns)
dfparams.loc[:,'param A'] = []
dfparams.loc[:,'param K'] = []
dfparams.loc[:,'param T'] = []
dfparams.loc[:,'RMSE'] = []
dfparams.loc[:,'R-squared'] = []

count=0
for name, group in gb1:
    count=count+1
    figure=plt.figure(fign)
    fign=fign+1
    
    Norman=pd.DataFrame({"DAF":[13],"Value":[0.018]})
    group=pd.concat([Norman,group])
    sigma = np.ones(len(group.DAF))
    sigma[[0]] = 0.018

    xnew = np.arange(group.DAF.min(), group.DAF.max())
    
    parameters, pcov = curve_fit(Gompertz2, group.DAF, group.Value,p0=[ 380, 0.035,73], sigma=sigma)#19.53383854,-50.41273115,  0.30155722, -0.88434497]
    #print (group.iloc[-1])
    
    group['param A']=parameters[0]
    group['param K']=parameters[1]
    group['param T']=parameters[2]
    
    
   
    y_gomp = Gompertz2(  xnew, parameters[0], parameters[1], parameters[2], )#parameters[3]
    modelPredictions = Gompertz2(group.DAF, parameters[0], parameters[1], parameters[2]) 

    absError = modelPredictions - group.Value
    
    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(group.Value))
    group['RMSE']=RMSE
    group['R-squared']=Rsquared
    dfparams = dfparams.append(group.iloc[-1])
    # print(name)
    # print('R-squared:', Rsquared)
    plt.plot(xnew,y_gomp,label=name, c='#8856a7')
    sns.scatterplot(data=dfiur,x=dfiur.DAF,y= dfiur.Value,hue= 'Reference')
    sns.scatterplot(data=group,x=group.DAF,y= group.Value,c='#8856a7')
    
    plt.legend(loc="lower right")
    plt.xlabel('DAF')
    plt.ylabel('Weitght (g)')
    plt.show()
    plt.savefig("fig/"+group.Strain_sex_auth.iloc[-1]+'.png')


print (',,,,,,,,,,,,'+str(count))
figure=plt.figure(fign)
fign=fign+1
plt.subplot(3, 1, 1)


dfparams[['Reference','Strain','Sex', 'RMSE', 'R-squared','param A', 'param K', 'param T']].to_excel("dfparams.xlsx")  

dfparams_no_OL=dfparams.drop([64,181])
bplot = sns.boxplot(y='param A', x='Strain_sex', 
                 data=dfparams_no_OL, 
                 width=0.5,
                 palette="colorblind")
plt.ylim(ymin=0)
bplot=sns.stripplot(y='param A', x='Strain_sex', 
                 data=dfparams_no_OL, 
                 palette="colorblind")
plt.ylim(ymin=0)
plt.subplot(3, 1, 2)
bplot = sns.boxplot(y='param K', x='Strain_sex', 
                 data=dfparams_no_OL, 
                 width=0.5,
                 palette="colorblind")
bplot=sns.stripplot(y='param K', x='Strain_sex', 
                 data=dfparams_no_OL, 
                 palette="colorblind")
plt.ylim(ymin=0)
plt.subplot(3, 1, 3)
bplot = sns.boxplot(y='param T', x='Strain_sex', 
                 data=dfparams_no_OL, 
                 width=0.5,
                 palette="colorblind")
bplot=sns.stripplot(y='param T', x='Strain_sex', 
                 data=dfparams, 
                 palette="colorblind")
plt.ylim(ymin=0)
plt.savefig("fig/"+'boxplot'+'.png')

df_OL=df[-df['Strain_sex_auth'].isin(['Wistar female Hernández-Álvarez 2019','SD male Alemáan 1998'])]
figure=plt.figure(fign)

fign=fign+1
parameters_All= dfparams_no_OL.describe(include='all').loc['mean']
print(parameters_All)
xnew_All = np.arange(0, 726)
yaj_All = Gompertz2(  xnew_All, parameters_All.loc['param A'],  parameters_All.loc['param K'], parameters_All.loc['param T'])


sns.scatterplot( data=df_OL,x='DAF', y='Value',style="Sex",alpha= 0.5)#, hue= 'Reference'
plt.plot(xnew_All,yaj_All,label='fit all datasets',c='#8856a7')#group.Strain_sex_auth.iloc[-1].legend(loc='upper left',ncol=2, title="Title")
plt.xlabel('DAF')
plt.ylabel('Body Weight (g)')
plt.savefig("fig/"+'GOMP_all_together'+'.png')


figure=plt.figure(fign)
fign=fign+1
gb2s = dfparams.groupby(['Strain'])

gb2all = gb2s.describe()

parameters_strain=gb2all.loc[:,['param A','param K','param T'] ].droplevel(0, axis=1) #('param a', 'mean')
parameters_strain=parameters_strain.loc[:,"mean"]
print (parameters_strain)#xnew_All = np.arange(0, 726)
#
yaj_SD = Gompertz2(  xnew_All,parameters_strain.iloc[1,0], parameters_strain.iloc[1,1], parameters_strain.iloc[1,2])
yaj_Wistar = Gompertz2(  xnew_All,parameters_strain.iloc[2,0], parameters_strain.iloc[1,1], parameters_strain.iloc[2,2])

#df_ND=df[df['Strain'] =='ND']
df_SD=df[df['Strain'] =='Sprague Dawley']
df_Wistar=df[df['Strain'] =='Wistar']

#sns.scatterplot( data=df_ND,x='DAF', y='Value',c=CB_color_cycle[0],alpha= 0.5)#, hue= 'Reference'
sns.scatterplot( data=df_OL,x='DAF', y='Value',hue='Strain', palette=palette_dict,style="Sex", legend=False,alpha= 0.5)

sns.lineplot (x=xnew_All,y=yaj_SD,label='SD',c=CB_color_cycle[1])#,dashes=True
sns.lineplot (x=xnew_All,y=yaj_Wistar,label='Wistar',c=CB_color_cycle[2])#,style="Sex"
plt.xlabel("DAF")
plt.ylabel("Body Weight (g)")
#print (gb2all)# mymodel = np.poly1d(np.polyfit(X, y, 8))
plt.savefig("fig/"+'GOMP_STARIN'+'.png')

figure=plt.figure(fign)
fign=fign+1
SD_sex= gb2s.get_group('Sprague Dawley').groupby(['Sex']).describe()
parameters_SD_sex=SD_sex.loc[:,['param A','param K','param T'] ].droplevel(0, axis=1) #('param a', 'mean')
parameters_SD_sex=parameters_SD_sex.loc[:,"mean"]
print (parameters_SD_sex)

yaj_SD_male = Gompertz2(  xnew_All,parameters_SD_sex.iloc[1,0], parameters_SD_sex.iloc[1,1],parameters_SD_sex.iloc[1,2])
yaj_SD_female = Gompertz2(  xnew_All,parameters_SD_sex.iloc[0,0], parameters_SD_sex.iloc[0,1],parameters_SD_sex.iloc[0,2])



sns.lineplot (x=xnew_All,y=yaj_SD_male,label='SD male',c=CB_color_cycle[1])
sns.lineplot (x=xnew_All,y=yaj_SD_female,label='SD female',c=CB_color_cycle[1],linestyle='dashed')
Wistar_sex= gb2s.get_group('Wistar').groupby(['Sex']).describe()
parameters_Wistar_sex=Wistar_sex.loc[:,['param A','param K','param T'] ].droplevel(0, axis=1) #('param a', 'mean')
parameters_Wistar_sex=parameters_Wistar_sex.loc[:,"mean"]

print (parameters_Wistar_sex)
yaj_Wistar_male = Gompertz2(  xnew_All,parameters_Wistar_sex.iloc[1,0], parameters_Wistar_sex.iloc[1,1],parameters_Wistar_sex.iloc[1,2])
yaj_Wistar_female = Gompertz2(  xnew_All,parameters_Wistar_sex.iloc[0,0], parameters_Wistar_sex.iloc[0,1],parameters_Wistar_sex.iloc[0,2])

sns.lineplot (x=xnew_All,y=yaj_Wistar_male,label='Wistar male',c=CB_color_cycle[2])
sns.lineplot (x=xnew_All,y=yaj_Wistar_female,label='Wistar female',c=CB_color_cycle[2],linestyle='dashed')
sns.scatterplot( data=df_OL,x='DAF', y='Value',hue='Strain', palette=palette_dict,style="Sex", legend=False,alpha= 0.5)

plt.ylabel('Body Weight (g)')
plt.savefig("fig/"+'GOMP_STARIN_sex'+'.png')
dfparams_no_OL=dfparams_no_OL.drop([275,270])#drop nd

modelo_mixto_A = smf.mixedlm("Q('param A') ~ C(Strain) +C(Sex)", 
                            dfparams_no_OL, groups=dfparams_no_OL['Reference'])

# Ajustar el modelo
resultado_A = modelo_mixto_A.fit()
# # get random effects


resid_A=resultado_A.resid
test_norm_A = sms.jarque_bera(resid_A)
test_hctt_A = sms.het_breuschpagan(resid_A , resultado_A.model.exog)
# print('test_norm_A')
# print(test_norm_A)
# print('test_hctt_A')
# print(test_hctt_A)
# print(resultado_A.random_effects)

print('------------------------------')
#Mostrar el resumen del modelo
print(resultado_A.summary())
modelo_mixto_K = smf.mixedlm("Q('param K') ~ C(Strain) +C(Sex)", 
                            dfparams_no_OL, groups=dfparams_no_OL['Reference'])

# Ajustar el modelo
resultado_K = modelo_mixto_K.fit()
# # get random effects
resid_K=resultado_K.resid
test_norm_K = sms.jarque_bera(resid_K)
test_hctt_K = sms.het_breuschpagan(resid_K , resultado_K.model.exog)
# print('test_norm_K')
# print(test_norm_K)
# print('test_hctt_K')
# print(test_hctt_K)
# print(resultado_K.random_effects)

print('------------------------------')
#Mostrar el resumen del modelo
# print(resultado_K.summary())
modelo_mixto_T = smf.mixedlm("Q('param T') ~ C(Strain) +C(Sex)", 
                            dfparams_no_OL, groups=dfparams_no_OL['Reference'])

# Ajustar el modelo
resultado_T = modelo_mixto_T.fit()
# # get random effects
resid_T=resultado_T.resid
test_norm_T = sms.jarque_bera(resid_T)
test_hctt_T = sms.het_breuschpagan(resid_T , resultado_T.model.exog)
# print('test_norm_T')
# print(test_norm_T)
# print('test_hctt_T')
# print(test_hctt_T)
# print(resultado_T.random_effects)

print('------------------------------')
#Mostrar el resumen del modelo
# print(resultado_T.summary())

with open('mixed model of parameters.txt', 'w') as f:
    f.write('param A')
    f.write(str(resultado_A.summary()))
    f.write('param K')
    f.write(str(resultado_K.summary()))
    f.write('param T')
    f.write(str(resultado_T.summary()))

