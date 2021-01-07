##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Other packages
from memory_reduction_script import reduce_mem_usage_sd as mr # I'm using a memory reduction script to reduce DataFrame memory size
from sklearn.impute import SimpleImputer

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")





##### Functions

# Create labels on charts
def autolabel(plot):
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center',
                       fontsize = 18,
                       xytext = (0, 9), 
                       textcoords = 'offset points') 
        
def countplot(column, values = True):
    plt.figure(figsize=(35, 10))
    count = df[column].dropna().value_counts()[:15]
    plot = sns.barplot(count.index, count.values)
    plt.title(f'{column} variability', fontsize = 18)
    max_value = max(df[column].value_counts())
    plot.set(ylim = (None,(max_value+max_value*0.1)), ylabel = None, yticklabels = [])
    plot.tick_params(left=False)
    if max([len(str(i)) for i in count.index]) >= 6:
        plt.xticks(rotation=90, fontsize = 16)
    else:
        plt.xticks(fontsize = 16)
    if values == True:
        autolabel(plot)
    plt.tight_layout()
    plt.show()





##### Import data
# Dataset has a major size: 13647309 rows x 48 columns. 
# I'm reducing memory with safe downcast function. Then, I restart the kernel and re-uploading it with 'memred' version to avoid memory crashes.

# Check the csv's path before running it
# Choose between 'Reducing memory' or 'Upload dataset with memory reduction'

action_1 = 'Reducing memory'
action_2 = 'Upload dataset with memory reduction'

select = action_2

if select == 'Reducing memory':
    data = pd.read_csv('train_ver2.csv.zip',dtype={"sexo":str,
                                                                                    "ind_nuevo":str,
                                                                                    "ult_fec_cli_1t":str,
                                                                                    "indext":str})

    data = mr(data, verbose = False)
    data.to_pickle('data_memred.pkl')
    
elif select == 'Upload dataset with memory reduction':
    df = pd.read_pickle('data_memred.pkl')





##### Brief statistic of the dataset
# Using safe downcast function, we reduce datatframe memory usage to 1gb only.

df.info()
df.describe()





##### Now I check variability of some columns

columns = list(df)
for delete in ['fecha_dato','ncodpers','pais_residencia','fecha_alta','renta']:
    columns.remove(delete)
    
for col in columns:
    print(f' {col} '.center(50,'#'))
    countplot(col, values = True)


# # Data Cleaning




##### Filtered by only currently active clients and from Spain
# I also filtered to obtain clients with information of whole 17 months (1 year and 4 months because 18th month
# is the one I'll try to predict) from users
# Number of persons ≠ number of transactions!

df = df.query('ind_actividad_cliente == 1')
df = df.query('pais_residencia == "ES"')
info_17 = df.ncodpers.value_counts()[df.ncodpers.value_counts()==17].index
df = df[df['ncodpers'].isin(info_17)].reset_index(drop = True)
df





##### Check % of NaNs of each column of the sataset

for col in df.columns:
    print(col, '=','{:.2%}'.format(df[col].isna().sum() / df.shape[0]))





##### Creation of two variables date-time related
# Customers are more likely to buy products at certain months of the year (seasonality)

df['fecha_dato'] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df['fecha_alta'] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")

df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["year_month"] = df["fecha_dato"].apply(lambda x: str(x.year) + '_' + str(x.month))





##### Two columns has a big number of NaNs that will cause NaNs-imputation problems so I delete them

print('Before:',df.shape)
for col in ['ult_fec_cli_1t','conyuemp']:
    del df[col]
print('After:',df.shape)





##### Then I delete rows where sexo, canal_entrada, segmento or nomprov values are NaN

print('Before:',df.shape)
for col in ['sexo','canal_entrada','segmento','nomprov']:
    df.dropna(axis = 0, subset = [col], inplace = True)
    
df.reset_index(drop = True, inplace = True)
print('After:',df.shape)





##### I delete some columns that don't seem relevant to the model

print('Before:',df.shape)
features_deleted = ["cod_prov",'ind_actividad_cliente',"indrel_1mes",'indresi',"tipodom","ind_empleado",
                    "pais_residencia","indrel","indext","indfall","ind_nuevo"]

df.drop(features_deleted, axis = "columns", inplace = True)
print('After:',df.shape)





##### I also delete the products that nobody buys

print('Before:',df.shape)
features_deleted = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cder_fin_ult1','ind_ctju_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
                    'ind_pres_fin_ult1','ind_viv_fin_ult1']

df.drop(features_deleted, axis = "columns", inplace = True)
print('After:',df.shape)





##### Replace of missing values using most frequent imputation strategy

print(f'Nº of missing values before replacement: {df.isna().sum().sum()}')

col_imp = [col for col in df.columns if df[col].isna().sum() != 0]
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
for col in col_imp:
    df[col] = imputer.fit_transform(df[[col]])

print(f'Nº of missing values after replacement: {df.isna().sum().sum()}')





##### Due to imputation strategy, I reapply previous filter to prevent unexpected values 

info_17 = df.ncodpers.value_counts()[df.ncodpers.value_counts()==17].index
df = df[df['ncodpers'].isin(info_17)].reset_index(drop = True)
df





##### Change of dtype of numeric columns to int32

int_col = ['age','ind_cco_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
           'ind_fond_fin_ult1','ind_hip_fin_ult1', 'ind_plan_fin_ult1','ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1','ind_nomina_ult1',
           'ind_nom_pens_ult1','ind_recibo_ult1','antiguedad']

df[int_col] = df[int_col].astype('int32')





##### Filter to get clients without age outliers values

print('Before:',df.shape)
df = df.query('age >= 18 & age <= 100')
print('After:',df.shape)





##### Binning of canal_entrada
# Keep only the 3 biggest values and convert the rest to the same unique group 'Others'

countplot('canal_entrada', values = True)

canal_entrada_values = list(df.canal_entrada.unique())
for col in ['KAT','KFC','KHE']:
    canal_entrada_values.remove(col)
df.canal_entrada = df.canal_entrada.replace(canal_entrada_values, 'otros')

countplot('canal_entrada', values = True)


# # Brief Exploratory Data Analysis




##### Products popularity

products_name = [i for i in list(df) if i.startswith('ind_')]
products_count = dict()
for col in products_name:
    products_count[col] = df[col].sum(axis=0)
products_count = dict(sorted(products_count.items(), key = lambda item: item[1], reverse=True))

plt.figure(figsize=(15, 15))
plt.title('Products popularity', fontsize = 18)
plt.pie(products_count.values(), labels = products_count.keys(), explode = [0.05 for i in range(len(products_count))],
        autopct='%.1f%%', shadow = True, labeldistance = 1.07, startangle = 45, rotatelabels = False)
plt.show()





##### Age histogram to show client profile

plt.figure(figsize=(35, 10))
sns.distplot(df.age, axlabel = "Age", color = "y")
plt.title("Dispersion of Age", fontsize = 18)
plt.show()





##### Number of clients per month

data = df.groupby(['year_month','ncodpers']).size().reset_index()['year_month'].value_counts()
data_2015_name = [f'2015_{i}' for i in range(1,13)]
data_2015_value = [data.loc[f'2015_{i}'] for i in range(1,13)]
data_2016_name = [f'2016_{i}' for i in range(1,6)]
data_2016_value = [data.loc[f'2016_{i}'] for i in range(1,6)]
data_name = data_2015_name + data_2016_name
data_value = data_2015_value + data_2016_value

plt.figure(figsize=(35, 10))
plot = sns.lineplot(x = data_name, y = data_value, sort = False)
plt.title("Number of clients per month", fontsize = 18)
plt.show()





df = mr(df, verbose = False)
df.to_pickle('data_modelling_memred.pkl')
print('Done!')

