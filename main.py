import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# =====================================================================
# Шаг 1. Загрузка датасета и вывод на экран
# =====================================================================
FILE_PATH = 'delivery_dataset.csv'  

# Читаем файл (если CSV)
df = pd.read_csv(FILE_PATH)
# Если файл Excel, используй: df = pd.read_excel(FILE_PATH)

print("Шаг 1: Первые 5 строк датасета до обработки:")
print(df.head())
print("-" * 50)

# =====================================================================
# Шаг 2. Выбор нужных столбцов и разделение выборки
# =====================================================================

# 1. Называем целевую колонку
TARGET_COLUMN = 'Время доставки (в минутах)'

# 2. Удаляем ненужные столбцы (ID и Дату)
df = df.drop(columns=['Дата заказа (ГГГГ-ММ-ДД)', 'Шифр заказа (ID)'])

# Разделяем на 3 части (по заданию: 300 / 100 / 100 последних)
train_df = df.iloc[:300]
val_df = df.iloc[300:400]
test_df = df.iloc[400:500]

# Разделяем каждую часть на X (признаки) и y (ответы)
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_val = val_df.drop(columns=[TARGET_COLUMN])
y_val = val_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print(f"Оставшиеся столбцы для обучения (X): {list(X_train.columns)}")
print("-" * 50)

# =====================================================================
# Шаги 4 и 5. Обучение Ridge(solver='sparse_cg') и построение графиков
# =====================================================================
train_sse, train_mse, train_rmse = [], [], []
val_sse, val_mse, val_rmse = [], [], []

max_iterations = 20

for i in range(1, max_iterations + 1):
    model = Ridge(solver='sparse_cg', max_iter=i, random_state=42)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    mse_tr = mean_squared_error(y_train, pred_train)
    mse_v = mean_squared_error(y_val, pred_val)

    train_mse.append(mse_tr)
    val_mse.append(mse_v)

    train_rmse.append(np.sqrt(mse_tr))
    val_rmse.append(np.sqrt(mse_v))

    train_sse.append(mse_tr * len(y_train))
    val_sse.append(mse_v * len(y_val))

# Рисуем графики
plt.figure(figsize=(16, 5))

# SSE
plt.subplot(1, 3, 1)
plt.plot(range(1, max_iterations + 1), train_sse, label='Train SSE', color='blue')
plt.plot(range(1, max_iterations + 1), val_sse, label='Val SSE', color='orange')
plt.title('SSE (Сумма квадратов ошибок)')
plt.xlabel('Итерации')
plt.legend()
plt.grid(True)

# MSE
plt.subplot(1, 3, 2)
plt.plot(range(1, max_iterations + 1), train_mse, label='Train MSE', color='blue')
plt.plot(range(1, max_iterations + 1), val_mse, label='Val MSE', color='orange')
plt.title('MSE (Среднеквадратичная ошибка)')
plt.xlabel('Итерации')
plt.legend()
plt.grid(True)

# RMSE
plt.subplot(1, 3, 3)
plt.plot(range(1, max_iterations + 1), train_rmse, label='Train RMSE', color='blue')
plt.plot(range(1, max_iterations + 1), val_rmse, label='Val RMSE', color='orange')
plt.title('RMSE (Корень из MSE)')
plt.xlabel('Итерации')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================================
# Шаг 6 и 7. Тестирование и итоговый RMSE
# =====================================================================
final_model = Ridge(solver='sparse_cg', max_iter=1000, random_state=42)
final_model.fit(X_train, y_train)

y_test_pred = final_model.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print(f"Шаг 7: Финальное значение RMSE на тестовой выборке: {test_rmse:.4f} минут")