# импорт необходимых библиотек
from tqdm import tqdm
import mlflow, os, json
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


if os.getenv("CI") == "true":
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://localhost:5000")

# Загружаем данные
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] =  iris.target

label_col = 'target'
ds = Dataset(iris_df, label=label_col, cat_features=[])

suite = data_integrity()
result = suite.run(ds)
report_dict = result.to_json()

with open("deepchecks_report.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2, ensure_ascii=False)

# делим выборку
X_train, X_test, y_train, y_test = train_test_split(
    iris_df.iloc[:, :-1], iris_df["target"], stratify=iris_df["target"], test_size=0.2, random_state=42
)

label_col = 'target'

X_train[label_col] = y_train
X_test[label_col] = y_test

ds_train = Dataset(X_train, label=label_col, cat_features=[])
ds_test = Dataset(X_test, label=label_col, cat_features=[])

suite = train_test_validation()
result = suite.run(ds_train, ds_test)
report_dict = result.to_json()

with open("deepchecks_report_train_test.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2, ensure_ascii=False)

# создаём отчёт
report = Report(metrics=[DataDriftPreset()])

# запускаем анализ
report.run(reference_data=X_train, current_data=X_test)

# сохраняем в HTML
report.save_html("drift_report_evidently.html")

n_trees = range(1, 160, 10)

mlflow.set_experiment("Experiment_for_DevOps_DZ_5_py")

for n_tree in tqdm(n_trees):
    # Начинаем эксперимент
    with mlflow.start_run():
        # Гиперпараметр

        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)

        # Оценка
        y_pred = rf_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Логирование параметров и метрик
        mlflow.log_param("n_trees", n_tree)
        mlflow.log_metric("accuracy", acc)

        # Сохранение модели
        mlflow.sklearn.log_model(rf_clf, artifact_path="model")

print("Эксперимент завершён и записан в MLflow")
print(f"Accuracy: {acc:.4f}")





