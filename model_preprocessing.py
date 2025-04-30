import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib
from pathlib import Path


def preprocess_data(
    input_folder, output_folder, filename, scaler=None, fit_scaler=False
):
    """
    Предварительная обработка данных: чтение, масштабирование и сохранение.

    Параметры:
    ----------
    input_folder : str
        Папка с исходными данными
    output_folder : str
        Папка для сохранения обработанных данных
    filename : str
        Имя файла с данными
    scaler : StandardScaler, optional
        Объект scaler (если None, создается новый)
    fit_scaler : bool, optional
        Нужно ли обучать scaler (по умолчанию False)

    Возвращает:
    -----------
    StandardScaler
        Обученный объект scaler
    """
    # Создаем папки, если они не существуют
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Чтение данных
    filepath = os.path.join(input_folder, filename)
    data = pd.read_csv(filepath)

    # Проверка наличия столбца temperature
    if "temperature" not in data.columns:
        raise ValueError(f"Column 'temperature' not found in {filename}")

    # Инициализация scaler
    if scaler is None:
        scaler = StandardScaler()

    # Масштабирование
    if fit_scaler:
        data["temperature"] = scaler.fit_transform(data[["temperature"]])
    else:
        data["temperature"] = scaler.transform(data[["temperature"]])

    # Сохранение обработанных данных
    processed_filename = filename.replace(".csv", "_processed.csv")
    output_path = os.path.join(output_folder, processed_filename)
    data.to_csv(output_path, index=False)

    return scaler


def main():
    # Обработка обучающих данных с обучением scaler
    train_scaler = preprocess_data(
        input_folder="train",
        output_folder="processed_data",
        filename="train_data.csv",
        fit_scaler=True,
    )

    # Сохранение scaler
    Path("models").mkdir(exist_ok=True)
    joblib.dump(train_scaler, "models/scaler.joblib")

    # Обработка тестовых данных с использованием обученного scaler
    preprocess_data(
        input_folder="test",
        output_folder="processed_data",
        filename="test_data.csv",
        scaler=train_scaler,
    )


if __name__ == "__main__":
    main()
