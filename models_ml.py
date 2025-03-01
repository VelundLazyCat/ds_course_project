import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Завантажимо файл і подивимось на вміст
path = r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\internet_service_churn.csv'


best_params_log = {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
best_params_tree = {'max_depth': None, 'max_features': 'sqrt',
                    'min_samples_leaf': 1, 'n_estimators': 200}
best_params_sgd = {'alpha': 0.1, 'loss': 'hinge', 'penalty': None}
accuracy_models = dict()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc


def get_data(path):
    data = pd.read_csv(path)
    # Згідно рекомендацій попередьної частини зробим потрібні перетворення по  очищенню данних датасета
    data = data.fillna(
        {'reamining_contract': data['reamining_contract'].min()}).dropna()
    data = data.iloc[:, 1:]
    print(data.isnull().sum())

    X = data.iloc[:, :-1]
    y = data.churn
    return X, y


def grid_search(X, y):

    param_grid = {'solver': ['lbfgs', 'sag', 'saga', 'liblinear'], "C": [
        100], "penalty": ['none', 'l1', 'l2', 'elasticnet']}
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
    logreg_cv.fit(X, y)

    # Отже найкращі гіперпараметри такі
    print("Найкращі гіперпараметри: ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)
    return logreg_cv.best_params_


def get_logreg_model(X_train, y_train, X_test, y_test, best_params):

    # Натренуємо модель зі зменшеними ознаками і обранами параметрами
    model = LogisticRegression(**best_params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Оцінка моделі
    log_5f_accuracy, log_5f_precision, log_5f_recall, log_5f_f1, log_5f_roc_auc = evaluate_model(
        model, X_test, y_test)
    print("Логистична регресія:")
    print(f"Точність: {log_5f_accuracy:.4f}, Повнота: {log_5f_recall:.4f}, F1-міра: {log_5f_f1:.4f}, ROC-AUC: {log_5f_roc_auc:.4f}")
    # порівнюючи з попередніми значеннями бачимо що показники погіршилися дуже незначно,
    # в 3-ому знаку після коми.
    accuracy_models['logreg'] = round(log_5f_accuracy, 4)

    return model


def get_randomtrees_model(X_train, y_train, X_test, y_test, best_params):
    # створення Random Forest класифікатора
    model = RandomForestClassifier(**best_params, random_state=42)
    # навчання моделі
    model.fit(X_train, y_train)

    print("Тренувальна виборка, score:", model.score(X_train, y_train))
    print("Тестова виборка, score:", model.score(X_test, y_test))
    model_tree_accuracy, model_tree_precision, model_tree_recall, model_tree_f1, model_tree_roc_auc = evaluate_model(
        model, X_test, y_test)
    print("Ліс випадкових дерев:")
    print(f"Точність: {model_tree_accuracy:.4f}, Повнота: {model_tree_precision:.4f}, F1-міра: {model_tree_f1:.4f}, ROC-AUC: {model_tree_roc_auc:.4f}")
    accuracy_models['ranfree'] = round(model_tree_accuracy, 4)
    return model


def get_sgd_model(X_train, y_train, X_test, y_test, best_params):
    model = SGDClassifier(**best_params, random_state=42)
    # навчання моделі
    model.fit(X_train, y_train)
    print("Тренувальна виборка, score:", model.score(X_train, y_train))
    print("Тестова виборка, score:", model.score(X_test, y_test))
    model_sgd_accuracy, model_sgd_precision, model_sgd_recall, model_sgd_f1, model_sgd_roc_auc = evaluate_model(
        model, X_test, y_test)
    print("Стохастичний градієнтний  спуск:")
    print(f"Точність: {model_sgd_accuracy:.4f}, Повнота: {model_sgd_precision:.4f}, F1-міра: {model_sgd_f1:.4f}, ROC-AUC: {model_sgd_roc_auc:.4f}")
    accuracy_models['sgd'] = round(model_sgd_accuracy, 4)
    return model

# пересвідчимось що classification_report надає нам те саме


if __name__ == '__main__':
    X, y = get_data(path)

    # Проведемо одразу нормалізацію вхідних даних
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape,
          y_train.shape, y_test.shape)

    # best_params = grid_search(X, y)

    model_logreg = get_logreg_model(
        X_train, y_train, X_test, y_test, best_params_log)

    model_tree = get_randomtrees_model(
        X_train, y_train, X_test, y_test, best_params_tree)

    model_sgd = get_sgd_model(
        X_train, y_train, X_test, y_test, best_params_sgd)

    estimators = [
        ('trees',  RandomForestClassifier(**best_params_tree, random_state=42)),
        ('logreg', LogisticRegression(**best_params_log, random_state=42)),
        ('sgd',   SGDClassifier(**best_params_sgd, random_state=42))
    ]
    final_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression())
    final_model.fit(X_train, y_train)
    print("Тренувальна виборка, score:", final_model.score(X_train, y_train))
    print("Тестова виборка, score:", final_model.score(X_test, y_test))
    model_accuracy, model_precision, model_recall, model_f1, model_roc_auc = evaluate_model(
        final_model, X_test, y_test)
    print("Стекінг:")
    print(f"Точність: {model_accuracy:.4f}, Повнота: {model_precision:.4f}, F1-міра: {model_f1:.4f}, ROC-AUC: {model_roc_auc:.4f}")
    accuracy_models['stack'] = round(model_accuracy, 4)

    print(accuracy_models)

    # joblib.dump(
    # model_logreg, r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\model_log.pkl')
    # joblib.dump(
    # model_tree, r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\model_tree.pkl')
    # joblib.dump(
    # model_sgd, r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\model_sgd.pkl')
    joblib.dump(
        final_model, r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\stacking_model_ml.pkl')
    joblib.dump(
        accuracy_models, r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\accuracy_ml.pkl')
