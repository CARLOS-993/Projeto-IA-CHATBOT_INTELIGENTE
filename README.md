# Projeto IA Chatbot - UNIC

Este é um projeto de **chatbot inteligente** desenvolvido em Python. Ele utiliza técnicas de Processamento de Linguagem Natural (NLP) para identificar intenções e responder de forma adequada.  

O projeto inclui scripts para **treinamento do modelo**, **pré-processamento de dados**, **revisão e re-treinamento** e **interface gráfica** para testar o chatbot.

## **Estrutura do projeto**

Projeto-IA-CHATBOT_INTELIGENTE/
│
├── chatbot_GUI.py # Interface gráfica do chatbot
├── preprocess.py # Pré-processamento de dados
├── chatbot_train_advanced.py # Treinamento do modelo
├── Review_and_retrain.py # Re-treinamento após revisão de dados
├── generate_metrics_csv.py # Geração de métricas em CSV
├── intents.json # Dados de intenções do chatbot
├── requirements.txt # Dependências Python
├── .gitignore # Arquivos/pastas ignoradas no Git
└── .venv/ # Ambiente virtual (não enviado ao GitHub)

## PASSOS PARA EXECUTAR O PROJETO
1. ## Clonar o repositório na sua Máquina
git clone https://github.com/CARLOS-993/Projeto-IA-CHATBOT_INTELIGENTE.git
cd Projeto-IA-CHATBOT_INTELIGENTE

## 2.Ativar o Ambiente Virtual

python -m venv .venv
.venv\Scripts\Activate.ps1

## 3. Instalar as dependências

pip install -r requirements.txt

## 4. Treinar o Modelo ; Testar o Modelo ; Ré-treinar o Modelo.

python chatbot_train_advanced.py

python chatbot_GUI.py

python Review_and_retrain.py



