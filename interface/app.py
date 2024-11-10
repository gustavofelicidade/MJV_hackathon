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
    st.error("Token de API não encontrado. Verifique se a variável de ambiente LLAMA_API_TOKEN está definida corretamente.")
    st.stop()

# Inicializar a API do Llama
llama = LlamaAPI(llama_token)

# Função para interagir com o Llama usando a biblioteca oficial
def llama_chatbot(message):
    # Construir a requisição para a API
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            # Adiciona um prompt de sistema em português
            {"role": "system", "content": "Você é um Assistente de Enfermagem de Triagem que responde em Português "
                                          "Brasileiro. Suas respostas devem ser breves e objetivas."},
            {"role": "user", "content": message},
        ],
        "functions": [
            {
                "name": "Assistente de Enfermagem de Triagem",
                "description": "Tem como objetivo otimizar a triagem de pacientes em departamentos de emergência",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "triagem": {
                            "type": "string",
                            "description": "Como um assistente profissional de enfermagem, você deve fornecer uma triagem",
                        },
                        "diagnostico": {
                            "type": "string",
                            "description": "O que você diagnostica com base na entrada do usuário",
                        }
                    },
                    "required": ["triagem", "diagnostico"],
                },
            }
        ],
        "stream": False,
        "function_call": "get_current_diagnosis",
        # Limita o número de tokens na resposta
        "max_tokens": 150,  # Ajuste este valor conforme necessário
        "temperature": 0.7,  # Opcional: controla a criatividade da resposta
    }

    # Executar a requisição
    response = llama.run(api_request_json)
    response_json = response.json()

    # Extrair a resposta do assistente
    assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', "Erro: Resposta não encontrada.")

    return assistant_message

# Função para limpar o histórico de chat
def clear_chat_history():
    st.session_state.messages = []

class Main:
    def __init__(self):
        # Título principal do app
        st.title("MJV Saúde")

        # Adicionar opções de navegação no sidebar
        page = st.sidebar.radio("Navegação", ["Triagem de Pacientes", "Assistente Virtual"])

        # Mostrar o botão 'Limpar Chat' no sidebar somente na página 'Assistente Virtual'
        if page == "Assistente Virtual":
            # Botão para limpar o chat no sidebar
            st.sidebar.button("Limpar Chat", on_click=clear_chat_history)

        if page == "Triagem de Pacientes":
            st.header("Triagem de Pacientes")

            # Configurando o upload de arquivo
            uploaded_file = st.file_uploader("Escolha um arquivo CSV com dados de pacientes para triagem",
                                             type="csv")

            # Configurando layout em duas colunas
            col1, col2 = st.columns(2)

            # Coluna 1: Exibição da imagem
            with col1:
                image_path = Path("./assets/images/image2.png")
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
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
                else:
                    st.info("Por favor, carregue um arquivo CSV para visualizar os dados.")

        elif page == "Assistente Virtual":
            st.header("Assistente Virtual")

            # Inicializar histórico de mensagens
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            # Exibir histórico de mensagens
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])

            # Campo de entrada para o chatbot
            user_input = st.chat_input("Digite sua mensagem para o assistente:")

            if user_input:
                # Adicionar mensagem do usuário ao histórico
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Exibir mensagem do usuário
                with st.chat_message("user"):
                    st.write(user_input)

                # Obter resposta do assistente
                with st.chat_message("assistant"):
                    with st.spinner("Assistente está digitando..."):
                        response = llama_chatbot(user_input)
                        st.write(response)
                        # Adicionar resposta ao histórico
                        st.session_state.messages.append({"role": "assistant", "content": response})

    # Você pode adicionar mais métodos ou funções à classe Main, se necessário

# Execução do aplicativo principal
if __name__ == "__main__":
    main_app = Main()
