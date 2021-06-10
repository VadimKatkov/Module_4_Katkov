# Этот файл явялется неотьемлемой частью системы файлов по модулю4 "Компьютер говорит нет"
# Файл используется для разработки модели, в то время как все визуализации выполняются в Jupyter Notebook

import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', None)

bank_set = pd.read_csv(r"C:\STUDY\SkillFactory\Module_4\Kaggle\train.csv", parse_dates=['app_date'])


def bin_labeling(data_set, columns_list):
    # функция-кодировщик бинарных признаков. передаем весь дата_сет и список бинарных полей
    for column in columns_list:
        data_set[column] = LabelEncoder().fit_transform(data_set[column])
    return data_set


def my_model_metrics(Y_test, y_predict):
    # функция расчета и вывода метрик
    print('--confusion_matrix--\n', confusion_matrix(Y_test, y_predict))
    print('---accuracy score--:', round(accuracy_score(Y_test, y_predict), 3))
    print('--precision score--:', round(precision_score(Y_test, y_predict), 3))
    print('---recall score----:', round(recall_score(Y_test, y_predict), 3))
    print('-----f1 score------:', round(f1_score(Y_test, y_predict), 3))
    print('---roc_auc score---:', round(roc_auc_score(Y_test, y_predict), 3))


# сгруппируем колонки по типам. мне понравился такой подход. удобно
cat_columns = ['home_address', 'work_address', 'first_time', 'sna']
bin_columns = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
num_columns = ['age', 'score_bki', 'bki_request_cnt', 'income', 'decline_app_cnt']

# сразу удалим client_id чтобы не мешался
bank_set.drop('client_id', axis=1, inplace=True)

# score_bki сдвинем в "положительную" часть оси, не все модели и методы принимают отрицательные и нулевые значения
bank_set['score_bki'] = list(map(lambda item: item + 4, bank_set['score_bki']))

# блок заполнения проусков в education при помощи MultinomialNB()
education_train = bank_set[num_columns][bank_set['education'].notna()]
education_target = bank_set['education'][bank_set['education'].notna()]
education_test = bank_set[num_columns][bank_set['education'].isna()]
naive_model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(education_train, education_target)
education_predict = naive_model.predict(education_test)
bank_set['education'][bank_set['education'].isna()] = education_predict

# По названиям категорий education видим, что они имеют упорядоченный характер
# (SCHool, UnderGRaduate, GRaDuate, PostGRaduate , ACaDemic). Сделаем принудительный labeling
education_dict = {'SCH': 0, 'UGR': 1, 'GRD': 2, 'PGR': 3, 'ACD': 4}
bank_set['education'] = bank_set['education'].map(education_dict)

# кодируем бинарные  и категориальные признаки
bank_set = bin_labeling(bank_set, bin_columns)
bank_set = pd.get_dummies(bank_set, columns=cat_columns)

# посчитаем кол-во дней 'app_date' до сегодняшнего дня и удалим признак 'app_date'
bank_set['days_past'] = list(map(lambda item: (pd.Timestamp(dt.date.today()) - item).days
                                 , bank_set['app_date']))
bank_set_v2 = bank_set.drop(columns=['app_date', 'age', 'sex', 'car',
                                     'work_address_1', 'work_address_2', 'work_address_3'])

# Обучаем модель и ищем Гиперпараметры
for column in bank_set_v2[['income', 'days_past']].columns:
    bank_set_v2[column] = bank_set_v2[column].apply(lambda item: np.log(abs(item) + 1))

X_grid = bank_set_v2.drop('default', axis=1)
Y_grid = bank_set_v2['default']

sampler = SMOTE()  # oversampling - тут были лучшие показатели
X, Y = sampler.fit_resample(X_grid, Y_grid)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,
                                                    test_size=0.3, shuffle=True)
myScaler = StandardScaler()
X_train = myScaler.fit_transform(X_train)
X_test = myScaler.transform(X_test)

my_Model = LogisticRegression(max_iter=10000)

param_grid = [
    {'penalty': ['l1'],
     'solver': ['liblinear', 'saga'],
     'class_weight': [None, 'balanced'],
     'multi_class': ['auto', 'ovr']},
    {'penalty': ['l2'],
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
     'class_weight': [None, 'balanced'],
     'multi_class': ['auto', 'ovr']},
    {'penalty': [None],
     'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
     'class_weight': [None, 'balanced'],
     'multi_class': ['auto', 'ovr']}]


searcher = GridSearchCV(my_Model, param_grid, scoring='roc_auc', cv=5, verbose=0)
best_model = searcher.fit(X_train, Y_train)
print('\nТест №10: гиперпараметры')
print('лучшие параметры:', best_model.estimator.get_params(), '\nBEST SCORE:', round(best_model.best_score_, 3))
print('встроеные score:', searcher.score(X_test, Y_test))

'''
Тест №10: гиперпараметры
лучшие параметры: 
'C': 1.0, 
'class_weight': None, 
'dual': False, 
'fit_intercept': True, 
'intercept_scaling': 1, 
'l1_ratio': None, 
'max_iter': 10000, 
'multi_class': 'auto', 
'n_jobs': None, 
'penalty': 'l2', 
'random_state': None, 
'solver': 'lbfgs', 
'tol': 0.0001, 
'verbose': 0, 
'warm_start': False} BEST SCORE: 0.803
встроеные score: 0.8042962458358729

Process finished with exit code 0'''