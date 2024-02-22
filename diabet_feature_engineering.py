import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('feature_engineering/feature_engineering/diabetes/diabetes.csv')
df.head()

df.shape
df.describe().T
df.columns
#eksik değer yok
df.isnull().sum()
df.info()
#Adım 1
def check_df(dataframe, head=2):
    print("############### Shape ######################################")
    print(dataframe.shape)
    print("############### Types ###############")
    print(dataframe.dtypes)
    print("############### Head ###############")
    print(dataframe.head(head))
    print("############### Tail ###############")
    print(dataframe.tail(head))
    print("############### NA ###############")
    print(dataframe.isnull().sum())
    print("############### Quantiles ###############")
    print(dataframe.describe([0, 0.05, 0.95, 0.99, 1]).T)

check_df(df)

#Adım 2 - Numerik ve kategorik değişkenleri yakalayınız.
#Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
def grab_col_names(dataframe, cat_th = 10, car_th = 20 ):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisindedir.

    """

    #-----------------------------------KATEGORİK DEĞİŞKENLERİN AYARLANMASI-------------------------------------
    #tipi kategori yada bool olanları seç
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    #integer yada float olup 10dan küçü değerdekileri yakala. Bu veriler numerik ama kategorik kaydedilir.
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    #tipleri kategorik ya da object olup eşsiz sınıf sayısı 20den fazla olanları getirç
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #-----------------------------------NUMERİK DEĞİŞKENLERİN AYARLANMASI-------------------------------------
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float]"]]
    num_cols = [col for col in df.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()
def cat_summary(dataFrame, col_name):
    print(pd.DataFrame({col_name: dataFrame[col_name].value_counts(),
                        "Ratio": 100 * dataFrame[col_name].value_counts() / len(dataFrame)}))
    print('###########################')


for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col,):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

#8 tane numerik col
for col in num_cols:
    num_summary(df, col)


#Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)

#hedef değişken analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.columns:
    print(str(col) + " " + str(check_outlier(df, col)))

#korelasyon analizi
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

#eksik değer analizi: eksik değer yok
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)


exclude_columns = ['Pregnancies', 'Outcome','Age']

zero_columns = [col for col in df.columns if col not in exclude_columns and (df[col] == 0).any()]

for n in zero_columns:
    df[n].replace(0, np.nan, inplace=True)


#eksik değer analizi: artık var
def missing_values_table(dataframe, na_name=False):
    #boş değere sahip kolonlar
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    #eksik değer sayısı
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    #eksik değer oranı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    #bunları df'e çevir
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

[col for col in df.columns if col not in exclude_columns]
#eksik değerleri ortalama ve medyan değerleriyle doldurduk.
df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
df["Glucose"].fillna(df["Glucose"].median(), inplace=True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace=True)


#eksik değer sorunu çözüldü.
missing_values_table(df, True)

#aykırı değer sorunu
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#aykırı değer var mı? var.
for col in num_cols:
    print(col, check_outlier(df, col))

#baskılama işlemi
for col in num_cols:
    replace_with_thresholds(df, col)

#baskılandı mı? tekrar aykırı değer var mı kontrolü
for col in num_cols:
    print(col, check_outlier(df, col))

#aykırı değer sorunu çözüldü.

#----------------------------------------------------------
#Yeni değişkenler oluşturunuz.

#age-level

df.loc[(df['Age'] > 21) & (df['Age'] < 31), 'NEW_AGE_CAT'] = 'young'

df.loc[(df['Age'] > 30) & (df['Age'] < 50), 'NEW_AGE_CAT'] = 'mature'

df.loc[(df['Age'] > 49) & (df['Age'] < 67), 'NEW_AGE_CAT'] = 'senior'

#BMI level
df.loc[(df['BMI'] < 18), 'NEW_BMI'] = 'underweight'

df.loc[(df['BMI'] > 18) & (df['BMI'] < 25), 'NEW_BMI'] = 'normalweight'

df.loc[(df['BMI'] > 26) & (df['BMI'] < 30), 'NEW_BMI'] = 'overweight'

df.loc[(df['BMI'] > 30) , 'NEW_BMI'] = 'obese'

df.head()
df['Age'].max()
df.groupby("SkinThickness")["Outcome"].mean()
df.groupby("BMI")["Outcome"].mean()

#------------------------------------------------------------
#Adım 3: Encoding işlemlerini gerçekleştiriniz

#binary col yok label encoder'a gerek yok
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df.nunique()
df = one_hot_encoder(df, ohe_cols)
df.head()
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)

#----------------------------------------------------------------------------
#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#--------------------------------------------------------------------------
# Yeni ürettiğimiz değişkenler ne alemde?
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
#--------------------------------------------------------------------------


df.loc[df['Age'] <= 30, 'NEW_AGE_CAT'] = 'young-age'
df.loc[(df['Age'] > 30) & (df['Age'] < 45), 'NEW_AGE_CAT'] = 'middle-age'
df.loc[df['Age'] > 45, 'NEW_AGE_CAT'] = 'old-age'
df.groupby('NEW_AGE_CAT').agg({'Outcome': 'mean'})

# RATIO_BMI/BP
df['RATIO_BMI/BP'] = df['BMI'] / df['BloodPressure']
df.groupby('Outcome').agg({'RATIO_BMI/BP': ['mean', 'count']})

# GLUCOSE_LEVEL
df['Glucose'].describe([0.25, 0.50, 0.75]).T
df.groupby('Outcome').agg({'Glucose': 'mean'})
df.loc[df['Glucose'] <= 99.750, 'GLUCOSE_LEVEL'] = 'Low'
df.loc[(df['Glucose'] > 99.750) & (df['Glucose'] <= 140.250), 'GLUCOSE_LEVEL'] = 'Mid'
df.loc[df['Glucose'] > 140.250, 'GLUCOSE_LEVEL'] = 'High'
df.head()

# PREG_AGE_SCORE
df['PREG_AGE_SCORE'] = df['Pregnancies'] * df['Age']
df.groupby('Outcome').agg({'PREG_AGE_SCORE': 'mean'})

# INS/SKT
df['INS/SKT'] = df['Insulin'] / df['SkinThickness']
df.groupby('Outcome').agg({'INS/SKT': 'mean'})

# AGE_BMI_div_SKT
df["AGE_BMI_div_SKT"] = (df['Age'] * df['BMI']) / df['SkinThickness']
df.groupby('Outcome').agg({'AGE_BMI_div_SKT': ['mean', 'count']})

# GLUCOSE_div_BP
df['GLUCOSE_div_BP'] = df['Glucose'] / df['BloodPressure']
df.groupby('Outcome').agg({'GLUCOSE_div_BP': 'mean'})

# BMI_HALF
df["BMI"].describe().T
df.loc[df["BMI"] <= 32.400, 'BMI_HALF'] = 'LOW_BMI'
df.loc[df["BMI"] > 32.400, 'BMI_HALF'] = 'HIGH_BMI'
df.groupby('BMI_HALF').agg({'Outcome': 'mean'})




zero_column = df.columns[(df == 0).any()]
