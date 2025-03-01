import streamlit as st
from functions import get_predictions, class_file_ext, class_accuracy

import pandas as pd


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


# Заголовок сторінки
st.title(":blue[Застосунок класифікації відтоку клієнтів]")

selectore = st.sidebar.selectbox(
    'Select', ["RandomForest Classificator",
               "Logistic regression Classificator",
               "SGD Classificator",
               "Ansamble ML Classificator",
               "Neural Network Classificator"])
st.info(
    f"Обрано {selectore}\n\rТочність класифікатора {class_accuracy[selectore]}%")

is_percent = st.sidebar.checkbox('Додати відсоток вирогідності прогнозу')

st.sidebar.header(
    "Завантажте csv-файл з данними клієнтів")
uploaded_file = st.sidebar.file_uploader(
    "Виберіть файл...", help='Прожмакай кнопку!', type=["csv", "CSV"])


if uploaded_file is not None:
    # Відкриття файла з данними
    try:
        predict_dataframe = False
        dataframe = pd.read_csv(uploaded_file)
        st.success(f"Завантажено файл: {uploaded_file.name}")

        st.write(dataframe)

        predict = st.sidebar.button(
            label='Зробити передбачення', key='predict')
        if predict:
            predict_dataframe = get_predictions(dataframe,
                                                model_choice=selectore,
                                                percents=is_percent)
            st.write("Датафрейм з передаченнями")
            st.write(predict_dataframe)

            csv_data = convert_df(predict_dataframe)
            file_name = f"{uploaded_file.name[:-4]}_{class_file_ext[selectore]}.csv"

            st.sidebar.download_button(
                label="Download data as CSV", data=csv_data, file_name=file_name, mime="text/csv")
            # st.sidebar.success(f"Передбачення збережені в файл: {file_name}")

    except Exception as e:
        st.error(f"ERROR:\n\r{e}")
