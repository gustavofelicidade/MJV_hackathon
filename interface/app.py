import streamlit as st
import pandas as pd
import json
from llamaapi import LlamaAPI
from pathlib import Path
from dotenv import load_dotenv
import os

# Carregar variáveis do arquivo .env
load_dotenv()
llama_token = os.getenv("LLAMA_API_TOKEN")

# Verificar se o token foi obtido corretamente
if not llama_token:
    st.error("Token de API não encontrado. Verifique se a variável de ambiente ENDPOINT está definida corretamente.")
    st.stop()

# Inicializar a API do Llama
llama = LlamaAPI(llama_token)

# Função para interagir com o Llama usando a biblioteca oficial
def llama_chatbot(message):
    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": message},
        ],
        "functions": [
            {
                "name": "Triage Nurse Assistant",
                "description": "aims to optimize patient triage in emergency departments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "triage": {
                            "type": "string",
                            "description": "As a professional Nurse Assistant you must provide a triage",
                        },
                        "diagnosis": {
                            "type": "number",
                            "description": "What you diagnose based on the user input",
                        }
                    },
                    "required": ["triage", "diagnosis"],
                },
            }
        ],
        "stream": False,
        "function_call": "get_current_diagnosis",
    }

    # Execute the Request
    response = llama.run(api_request_json)
    response_json = response.json()

    # Extrair a resposta do assistente
    assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', "Erro: Resposta não encontrada.")

    return assistant_message

class Main:
    def __init__(self):
        # Título no sidebar
        st.sidebar.title("Opções do MJV Saúde")
        st.sidebar.write("Use as opções abaixo para interagir com o aplicativo.")

        # Configurando o upload de arquivo no sidebar
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV com dados de pacientes para triagem",
                                                 type="csv")

        # Chatbot no sidebar
        st.sidebar.subheader("Assistente Virtual")

        # Campo de entrada para o chatbot
        user_input = st.sidebar.text_input("Digite sua mensagem para o assistente:")
        if user_input:
            response = llama_chatbot(user_input)
            st.sidebar.write("Resposta do Assistente:")
            st.sidebar.write(response)

        # Título principal do app
        st.title("Triagem de Pacientes - MJV Saúde")

        # Configurando layout em duas colunas
        col1, col2 = st.columns(2)

        # Coluna 1: Exibição da imagem
        with col1:
            image_path = Path("./assets/images/image2.png")
            if image_path.exists():
                st.image(str(image_path), use_column_width=True)
            else:
                st.warning("Imagem não encontrada em ./assets/images/image2.png")

        # Coluna 2: Exibição dos dados carregados (se houver um arquivo carregado)
        with col2:
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.write("Dados Carregados:")
                st.write(data.head())

                # Aqui você pode adicionar o código para carregar o modelo e fazer previsões
                # model = load_model("data/model/trained_model.pkl")
                # predictions = predict_critical_cases(model, data)

                st.write("Previsões:")
                # st.write(predictions)

# Execução do aplicativo principal
if __name__ == "__main__":
    main_app = Main()
