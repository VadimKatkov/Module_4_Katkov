# Этот файл явялется неотьемлемой частью системы файлов по модулю4 "Компьютер говорит нет"
# Файл используется для разработки модели, в то время как все визуализации выполняются в Jupyter Notebook

import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', None)

'''
client_id - идентификатор клиента
education - уровень образования
sex - пол заемщика
age - возраст заемщика
car - флаг наличия автомобиля
car_type - флаг автомобиля иномарки
decline_app_cnt - количество отказанных прошлых заявок
good_work - флаг наличия “хорошей” работы
bki_request_cnt - количество запросов в БКИ
home_address - категоризатор домашнего адреса
work_address - категоризатор рабочего адреса
income - доход заемщика
foreign_passport - наличие загранпаспорта
sna - связь заемщика с клиентами банка
first_time - давность наличия информации о заемщике
score_bki - скоринговый балл по данным из БКИ
region_rating - рейтинг региона
app_date - дата подачи заявки
default - флаг дефолта по кредиту
'''

bank_set = pd.read_csv(r"C:\STUDY\SkillFactory\Module_4\Kaggle\train.csv", parse_dates=['app_date'])
bank_set_check = pd.read_csv(r"C:\STUDY\SkillFactory\Module_4\Kaggle\test.csv", parse_dates=['app_date'])
bank_set_check['default'] = 99 # обозначит тестовый набор, чтобы потом его вычленить
bank_set = bank_set.append(bank_set_check, sort=False).reset_index(drop=True)


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


def model_prepare(data_set, scale_type=None):
    # функция для обучения модели
    # так как склеили два data_set теперь удалаем проверочные данные, которым в колонке default присвоили '99'
    X_base = data_set[data_set['default'] != 99].drop('default', axis=1)
    Y_base = data_set[data_set['default'] != 99]['default']

    # убираем дисбаланс в выборк. Пробуем разные виды семплинга
    # sampler = RandomOverSampler()  #oversampling
    # sampler = RandomUnderSampler()  # undersampling - тут были лучшие показатели
    sampler = SMOTE(random_state=0)  # SMOTE показал наилучшие результаты
    X, Y = sampler.fit_resample(X_base, Y_base)  # SMOTE сеплинг

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,
                                                        test_size=0.3, shuffle=True)

    my_Model = LogisticRegression(max_iter=5000, C=1)
    if scale_type == 'Standard':
        myScaler = StandardScaler()
        X_train = myScaler.fit_transform(X_train)
        X_test = myScaler.transform(X_test)
    elif scale_type == 'Robust':
        myScaler = RobustScaler()
        X_train = myScaler.fit_transform(X_train)
        X_test = myScaler.transform(X_test)
        my_Model.fit(X_train, Y_train)

    '''
    # тестирование MinMax показало результаты хуже. Думаю, что Standard и Robust лучше справляются с 
    # выбросами, которых у нас много и если будем их удалять - вырежем большую часть модели
    elif scale_type == 'MinMax':
        myScaler = MinMaxScaler()
        X_train = myScaler.fit_transform(X_train[['income', 'region_rating']])
        X_test = myScaler.transform(X_test[['income', 'region_rating']])
        my_Model.fit(X_train, Y_train)
    '''
    my_Model.fit(X_train, Y_train)

    return Y_test, my_Model.predict(X_test)


def outlayers(data_set, num_list):
    # функция обработки выбросов. применяем классический подход - 25%, 75% и +/-1.5 интерквартиля
    for column in data_set[num_list].columns:
        quantile_1 = np.quantile(data_set.loc[:, column], 0.25, interpolation='midpoint')
        quantile_3 = np.quantile(data_set.loc[:, column], 0.75, interpolation='midpoint')
        iqt = quantile_3 - quantile_1

        data_set.drop(data_set[(data_set[column] > quantile_3 + 1.5 * iqt) |
                               (data_set[column] < quantile_1 - 1.5 * iqt)].index, inplace=True)
    return data_set


# начинаем с чистки признаков. Визуализация признаков сделана в файле .ipynb этого пакета

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
bank_set.drop('app_date', axis=1, inplace=True)


# Тест №1: обращаемся к наивной модели
Y_test_v1, y_predict_v1 = model_prepare(bank_set)
print('Тест №1: метрики наивной модели')
my_model_metrics(Y_test_v1, y_predict_v1)

# Тест №2: Удалим коллинеарные признаки и наименее значимые
bank_set_v2 = bank_set.drop(columns=['age', 'sex', 'car', 'days_past',
                                     'work_address_1', 'work_address_2', 'work_address_3'])
Y_test_v2, y_predict_v2 = model_prepare(bank_set)
print('\nТес т№2: удалали часть признаков')
my_model_metrics(Y_test_v2, y_predict_v2)

# Тест №3: стандартизуем признаки при помощи StandardScaler()
Y_test_v3, y_predict_v3 = model_prepare(bank_set, 'Standard')
print('\nТес т№3: добавили стандартизацию Standard')
my_model_metrics(Y_test_v3, y_predict_v3)

# Тест №4: стандартизуем признаки при помощи RobustScaler()
Y_test_v4, y_predict_v4 = model_prepare(bank_set, 'Robust')
print('\nТест №4: добавили стандартизацию Robust')
my_model_metrics(Y_test_v4, y_predict_v4)

# Тест №5: удалим выбросы
bank_set_v5 = outlayers(bank_set, ['decline_app_cnt', 'bki_request_cnt', 'score_bki'])
Y_test_v5, y_predict_v5 = model_prepare(bank_set_v5)
print('\nТест №5: удалили выбросы')
my_model_metrics(Y_test_v5, y_predict_v5)

# Тест №6: логарифмирование числовых показателей
bank_set_v6 = bank_set.copy()
for column in bank_set_v6[['decline_app_cnt', 'bki_request_cnt', 'score_bki', 'income']].columns:
    bank_set_v6[column] = bank_set_v6[column].apply(lambda item: np.log(abs(item) + 1))
Y_test_v6, y_predict_v6 = model_prepare(bank_set_v6)
print('\nТест №6: логарифмирование числовых показателей')
my_model_metrics(Y_test_v6, y_predict_v6)


# Тест № FINAL: в этом блоке подставляем параметры полученные в сетке гиперпараметров
# проводим обучении на полном наборе train
for column in bank_set[['decline_app_cnt', 'bki_request_cnt', 'score_bki', 'income']].columns:
    bank_set[column] = bank_set[column].apply(lambda item: np.log(abs(item) + 1))

X_grid = bank_set[bank_set['default'] != 99].drop('default', axis=1)
Y_grid = bank_set[bank_set['default'] != 99]['default']

sampler = SMOTE()  # undersampling - тут были лучшие показатели
X, Y = sampler.fit_resample(X_grid, Y_grid)

myScaler = StandardScaler()
X_train = myScaler.fit_transform(X)

final_Model = LogisticRegression(C=1,
                                 class_weight=None,
                                 dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1,
                                 l1_ratio=None,
                                 max_iter=10000,
                                 multi_class='auto',
                                 n_jobs=None,
                                 penalty='l2',
                                 random_state=None,
                                 solver='lbfgs',
                                 tol=0.0001,
                                 verbose=0,
                                 warm_start=False)

final_Model.fit(X, Y)
print('\nТест № FINAL: митрики после GridSearch')
my_model_metrics(Y, final_Model.predict(X))

submision_set = bank_set[bank_set['default'] == 99].drop('default', axis=1)
prediction = pd.DataFrame(final_Model.predict_proba(submision_set)[:,1], columns=['default'])

# тут формируем файл submission для kaggle
submision_client_id = pd.read_csv(r"C:\STUDY\SkillFactory\Module_4\Kaggle\test.csv", parse_dates=['app_date'])
prediction['client_id'] = pd.DataFrame(submision_client_id['client_id'])
print(prediction.to_excel(r'C:\STUDY\SkillFactory\Module_4\Module_task\submision.xlsx'))
