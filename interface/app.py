import streamlit as st
import pandas as pd
from models.predict import predict_critical_cases
from utils.helpers import load_model
# from models import p

class Main:
    def __init__(self):
        st.title("Triagem de Pacientes - MJV Saúde ")

        uploaded_file = st.file_uploader("Escolha um arquivo CSV com dados de pacientes para triagem", type="csv")

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Dados Carregados:")
            st.write(data.head())

            model = load_model("data/model/trained_model.pkl")
            # predictions = predict_critical_cases(model, data)

            st.write("Previsões:")
            # st.write(predictions)
