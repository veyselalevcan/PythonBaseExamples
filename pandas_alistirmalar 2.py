def alternating_string(string):
    new_str = ''
    for indeksstr in range(len(string)):
        if indeksstr % 2 == 0:
            new_str += string[indeksstr].upper()
        else:
            new_str += string[indeksstr].lower()
    print(new_str)

alternating_string('veysel bunu deftere yaz unutma anlat')

def alterne_enumerate(string):
    new_str = ''
    for indeks, letter in enumerate(string, 1):
        if indeks % 2 == 0:
            new_str += letter.lower()
        else:
            new_str = new_str + letter.upper()
    print(new_str)

alterne_enumerate('veysel bunu enumerate ile yazdın farkına var')

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset('titanic')

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df['sex'].value_counts()


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df['pclass'].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[['pclass', 'parch']].nunique()


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df['embarked'].info
df['embarked'].dtypes
df['embarked'] = df['embarked'].astype('category')
df['embarked'].dtype
#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df['embarked'] == 'C'].tail()


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df['embarked'] != 'S'].head()

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################
df[(df['age'] < 30) & (df['sex'] == 'female')].head()


#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df['age'] > 70) | (df['fare'] > 500)].head()


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop('who', axis=1).head()
df.columns
#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

df['deck'].fillna(df['deck'].mode()[0], inplace=True)
df['deck'].isnull().sum()
df['deck'].value_counts()


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df['age'].isnull().sum()
df['age'].fillna(df['age'].median(), inplace=True)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(['pclass', 'sex']).agg({'survived': ['sum', 'count', 'mean']})

df.pivot_table('survived', ['pclass', 'sex'], aggfunc=['sum', 'count', 'mean'])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def func_age(age):
    if age < 30:
        return 1
    else:
        return 0


df['age_flag'] = [func_age(i) if i < 30 else func_age(i) for i in df['age']]

df['age_flag'] = df['age'].apply(lambda x: func_age(x))
df['age_flag'].value_counts()

df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df_tips= sns.load_dataset('tips')
df_tips.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df_tips.groupby('time').agg({'total_bill': ['sum', 'min', 'max', 'mean']})


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df_tips.groupby(['time', 'day']).agg({'total_bill': ['sum', 'min', 'max', 'mean']})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df_tips[(df_tips['time'] == 'Lunch') &
        (df_tips['sex'] == 'Female')].groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean'],
                                                          'tip': ['sum', 'min', 'max', 'mean']})


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df_tips[(df_tips['size'] < 3) & (df_tips['total_bill'] > 10)].mean()
df_tips.loc[(df_tips['size'] < 3) & (df_tips['total_bill'] > 10), 'total_bill'].mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df_tips['total_bill_tip_sum'] = df_tips['total_bill'] + df_tips['tip']
df_tips['total_bill_tip_sum'].head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

df_tips_new = df_tips['total_bill_tip_sum'].sort_values(ascending=False).head(30)
df_tips_new.shape

new = df_tips.sort_values('total_bill_tip_sum', ascending=False)[:30].head()
new.shape

new_1 = pd.DataFrame(df_tips['total_bill_tip_sum'].sort_values(ascending=False).head(30))
type(new_1)
type(new)
type(df_tips_new)