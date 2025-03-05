#!/usr/bin/env python
# coding: utf-8

# In[21]:


pip install plotly


# In[22]:


# Import the libraries required for exploration and preproccesing
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

reload(plt)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

warnings.filterwarnings('ignore')

# configure plotly graph objects
pio.renderers.default = 'iframe'
# pio.renderers.default = 'vscode'

pio.templates["ck_template"] = go.layout.Template(
    layout_colorway = px.colors.sequential.Viridis, 
#     layout_hovermode = 'closest',
#     layout_hoverdistance = -1,
    layout_autosize=False,
    layout_width=800,
    layout_height=600,
    layout_font = dict(family="Calibri Light"),
    layout_title_font = dict(family="Calibri"),
    layout_hoverlabel_font = dict(family="Calibri Light"),
#     plot_bgcolor="white",
)
 
# pio.templates.default = 'seaborn+ck_template+gridon'
pio.templates.default = 'ck_template+gridon'
# pio.templates.default = 'seaborn+gridon'
# pio.templates


# In[24]:


# Give names to the features
index_names = ['engine', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]
col_names = index_names + setting_names + sensor_names

# df_train = pd.read_csv(('./CMaps/train_FD001.txt'), sep='\s+', header=None, names=col_names)
# df_test = pd.read_csv(('./CMaps/test_FD001.txt'), sep='\s+', header=None, names=col_names)
# df_test_RUL = pd.read_csv(('./CMaps/RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

df_train = pd.read_csv(('train_FD001.txt'), sep='\\s+', header=None, names=col_names)
df_test = pd.read_csv(('test_FD001.txt'), sep='\\s+', header=None, names=col_names)
df_test_RUL = pd.read_csv(('RUL_FD001.txt'), sep='\\s+', header=None, names=['RUL'])


# In[25]:


df_train.head()


# In[26]:


df_test.head()


# In[27]:


df_train.info()


# In[28]:


df_train.describe(include='all').T


# In[29]:


plt.figure(figsize=(10,10))
threshold = 0.90
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster2 = df_train.corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster2,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')


# In[38]:


get_ipython().system('pip uninstall -y pandas-profiling pydantic pydantic-settings')


# In[39]:


get_ipython().system('pip install -U pandas-profiling pydantic-settings')


# In[42]:


pip install -U ydata-profiling


# In[43]:


from ydata_profiling import ProfileReport


# In[44]:


get_ipython().run_cell_magic('time', '', 'profile = ProfileReport(df_train,\n                        title="Predictive Maintenance",\n                        dataset={"description": "This profiling report was generated for Carl Kirstein",\n                                 "copyright_holder": "Carl Kirstein",\n                                 "copyright_year": "2022",\n                                },\n                        explorative=True,\n                       )\nprofile\n')


# In[45]:


# drop the sensors wiith constant values
sens_const_values = []
for feature in list(setting_names + sensor_names):
    try:
        if df_train[feature].min()==df_train[feature].max():
            sens_const_values.append(feature)
    except:
        pass

print(sens_const_values)
df_train.drop(sens_const_values,axis=1,inplace=True)
df_test.drop(sens_const_values,axis=1,inplace=True)


# In[47]:


# drop all but one of the highly correlated features
cor_matrix = df_train.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(corr_features)
df_train.drop(corr_features,axis=1,inplace=True)
df_test.drop(corr_features,axis=1,inplace=True)


# In[48]:


list(df_train)


# In[49]:


df_train.head()


# In[50]:


features = list(df_train.columns)


# In[51]:


# check for missing data
for feature in features:
    print(feature + " - " + str(len(df_train[df_train[feature].isna()])))


# In[52]:


# define the maximum life of each engine, as this could be used to obtain the RUL at each point in time of the engine's life 
df_train_RUL = df_train.groupby(['engine']).agg({'cycle':'max'})
df_train_RUL.rename(columns={'cycle':'life'},inplace=True)
df_train_RUL.head()


# In[53]:


df_train=df_train.merge(df_train_RUL,how='left',on=['engine'])


# In[55]:


df_train['RUL']=df_train['life']-df_train['cycle']
df_train.drop(['life'],axis=1,inplace=True)

# the RUL prediction is only useful nearer to the end of the engine's life, therefore we put an upper limit on the RUL
# this is a bit sneaky, since it supposes that the test set has RULs of less than this value, the closer you are
# to the true value, the more accurate the model will be
df_train['RUL'][df_train['RUL']>125]=125
df_train.head()


# In[62]:


plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams['figure.figsize']=8,40 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 0.1
plot_items = list(df_train.columns)[1:-1]
fig,ax = plt.subplots(len(plot_items),sharex=True)
ax[0].invert_xaxis()

engines = list(df_train['engine'].unique())

for engine in engines[10:30]:
    for i,item in enumerate(plot_items):
        f = sns.lineplot(data=df_train[df_train['engine']==engine],x='RUL',y=item,color='steelblue',ax=ax[i],
                        )


# In[63]:


from scipy import signal
def smooth_function(x,window=15,order=3):
    return signal.savgol_filter(x,window,order)


# In[64]:


# for engine in engines:
#     for item in sensor_names:
#         try:
#             df_train[item][df_train['engine']==engine]=smooth_function(df_train[item][df_train['engine']==engine])
#             df_test[item][df_test['engine']==engine]=smooth_function(df_test[item][df_test['engine']==engine])
#         except:
#             pass


# In[65]:


# plt.style.use('seaborn-white') 
# plt.rcParams['figure.figsize']=8,40 
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.size'] = 8
# plt.rcParams['lines.linewidth'] = 0.1
# plot_items = list(df_train.columns)[1:-1]
# fig,ax = plt.subplots(len(plot_items),sharex=True)
# ax[0].invert_xaxis()
# for engine in engines:
#     for i,item in enumerate(plot_items):
#         f = sns.lineplot(data=df_train[df_train['engine']==engine],x='RUL',y=item,color='steelblue',ax=ax[i],
#                         )


# # Preprocessing and feature selection

# In[66]:


# awesome bit of code from https://www.kaggle.com/code/adibouayjan/house-price-step-by-step-modeling

Selected_Features = []
import statsmodels.api as sm

def backward_regression(X, y, initial_list=[], threshold_out=0.05, verbose=True):
    """To select feature with Backward Stepwise Regression 

    Args:
        X -- features values
        y -- target variable
        initial_list -- features header
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling 
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")


# Application of the backward regression function on our training data
X = df_train.iloc[:,1:-1]
y = df_train.iloc[:,-1]
backward_regression(X, y)


# In[67]:


Selected_Features


# In[68]:


# X.head()
feature_names = Selected_Features[0]
np.shape(X)


# In[69]:


len(feature_names)


# # Modelling and Evaluation

# In[70]:


import time
model_performance = pd.DataFrame(columns=['r-Squared','RMSE','total time'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score

import sklearn
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor

model_performance = pd.DataFrame(columns=['R2','RMSE', 'time to train','time to predict','total time'])


def R_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


# In[71]:


df_test_cycle = df_test.groupby(['engine']).agg({'cycle':'max'})
df_test_cycle.rename(columns={'cycle':'life'},inplace=True)
df_test_max = df_test.merge(df_test_cycle,how='left',on=['engine'])
df_test_max = df_test_max[(df_test_max['cycle']==df_test_max['life'])]
df_test_max.drop(['life'],axis=1,inplace=True)
# df_test_max


# In[72]:


X_train = df_train[feature_names]
y_train = df_train.iloc[:,-1]
X_test = df_test_max[feature_names]
y_test = df_test_RUL.iloc[:,-1]


# In[73]:


X_train.head()


# In[74]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[76]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # kNN

# In[77]:


get_ipython().run_cell_magic('time', '', 'from sklearn.neighbors import KNeighborsRegressor\nstart = time.time()\nmodel = KNeighborsRegressor(n_neighbors=9).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\n\n\nmodel_performance.loc[\'kNN\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[79]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=5,5 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,150),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[80]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=20,5 

fig,ax = plt.subplots()
plt.ylabel('RUL')
plt.xlabel('Engine nr')

g = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_test,
                color='gray',
                label = 'actual',
                ax=ax)

f = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_predictions,
                color='steelblue',
                label = 'predictions',
                ax=ax)
ax.legend()


# # SVM

# In[81]:


get_ipython().run_cell_magic('time', '', 'from sklearn.svm import SVR\nstart = time.time()\nmodel = SVR(kernel="rbf", C=100, gamma=0.5, epsilon=0.01).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\nmodel_performance.loc[\'SVM\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[83]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=5,5 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,150),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[85]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=20,5 

fig,ax = plt.subplots()
plt.ylabel('RUL')
plt.xlabel('Engine nr')

g = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_test,
                color='gray',
                label = 'actual',
                ax=ax)

f = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_predictions,
                color='steelblue',
                label = 'predictions',
                ax=ax)
ax.legend()


# # Random Forest

# In[86]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\nstart = time.time()\nmodel = RandomForestRegressor(n_jobs=-1,\n                              n_estimators=500,\n                              min_samples_leaf=1,\n                              max_features=\'sqrt\',\n                             ).fit(X_train,y_train)\nend_train = time.time()\ny_predictions = model.predict(X_test) # These are the predictions from the test data.\nend_predict = time.time()\n\nmodel_performance.loc[\'Random Forest\'] = [model.score(X_test,y_test), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(model.score(X_test,y_test)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[87]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=5,5 

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,150),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[89]:


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize']=20,5 

fig,ax = plt.subplots()
plt.ylabel('RUL')
plt.xlabel('Engine nr')

g = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_test,
                color='gray',
                label = 'actual',
                ax=ax)

f = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_predictions,
                color='steelblue',
                label = 'predictions',
                ax=ax)
ax.legend()


# # LSTM

# In[91]:





# In[97]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import sklearn


# In[98]:


model = keras.Sequential()
model.add(LSTM(100,  
                return_sequences=True,
               input_shape=(1,X_train.shape[1])
              ))
model.add(BatchNormalization())
model.add(LSTM(50,
                return_sequences=True,
               activation='tanh'
              ))
model.add(Dropout(0.5))
model.add(LSTM(10,
               return_sequences=True,
               activation='tanh',
              ))
model.add(Dropout(0.5))
model.add(Dense(100,
               activation='relu',
              ))
model.add(Dense(1))


# In[99]:


initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)


# In[100]:


model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))


# In[101]:


model.summary()


# In[102]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=4, min_lr=1e-7, verbose=1)


# In[103]:


from sklearn.model_selection import train_test_split

X_train_s, X_val, y_train_s, y_val = train_test_split(X_train, y_train, test_size=0.1)

#The LSTM input layer must be 3D.
#The meaning of the 3 input dimensions are: samples, time steps, and features.
#reshape input data
X_train_reshaped = X_train_s.reshape(X_train_s.shape[0], 1, X_train_s.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[104]:


get_ipython().run_cell_magic('time', '', '\nstart = time.time()\nhistory = model.fit(x=X_train_reshaped,y=y_train_s,\n                    validation_data = (X_val_reshaped,y_val),\n                    epochs = 30,\n                    # shuffle = True,\n                    batch_size = 500,\n                    callbacks=[reduce_lr]\n                   )\nend_train = time.time()\ny_predictions = model.predict(X_test_reshaped) # These are the predictions from the test data.\nend_predict = time.time()\ny_predictions = y_predictions[:,0][:,0]\nmodel_performance.loc[\'LSTM\'] = [sklearn.metrics.r2_score(y_test, y_predictions), \n                                   mean_squared_error(y_test,y_predictions,squared=False),\n                                   end_train-start,\n                                   end_predict-end_train,\n                                   end_predict-start]\n\nprint(\'R-squared error: \'+ "{:.2%}".format(sklearn.metrics.r2_score(y_test, y_predictions)))\nprint(\'Root Mean Squared Error: \'+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))\n')


# In[106]:


plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.figsize']=5,5 

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=100,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(sklearn.metrics.r2_score(y_test, y_predictions)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,150),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


# In[107]:


plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.figsize']=20,5 

fig,ax = plt.subplots()
plt.ylabel('RUL')
plt.xlabel('Engine nr')

g = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_test,
                color='gray',
                label = 'actual',
                ax=ax)

f = sns.lineplot(x = np.arange(0,len(df_train['engine'].unique())),
                y=y_predictions,
                color='steelblue',
                label = 'predictions',
                ax=ax)
ax.legend()


# In[110]:


plt.style.use('seaborn-v0_8-white') 
plt.rcParams['figure.figsize']=5,5 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1.5
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# # Evaluate

# In[111]:


model_performance.style.background_gradient(cmap='RdYlBu_r').format({'R2': '{:.2%}',
                                                                     'RMSE': '{:.2f}',
                                                                     'time to train':'{:.3f}',
                                                                     'time to predict':'{:.3f}',
                                                                     'total time':'{:.3f}',
                                                                     })


# # conclusion

# The models do not seem that accurate, but the remaining useful life (RUL) when less than 50 gets predicted far more accurately.
# 
# Surprisingly the models are more accurate when the data is not smoothed. It seems that the noisyness of the sensors does carry enough information for the predictions to remain accurate.
# 
# A pity of the modelling is that the time leading up to the final points in the test data is lost. It carries potentially more information to make the model even more accurate if the timeseries prediction for LSTM is done.
# 
# I prefer the kNN and the Random Forest models. They seem the most robust (insensitive to how the models are set up). The LSTM model especially is difficult to tune, it takes long to train, and the prediction speed is not great when compared to the others.

# In[ ]:




