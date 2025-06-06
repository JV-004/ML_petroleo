# Sistema de Classificação da Qualidade da Água

Este projeto é um sistema de machine learning que classifica a qualidade da água com base em parâmetros físico-químicos, identificando contaminação por óleo. Ele oferece uma interface interativa para treinamento do modelo, predição em tempo real e análise detalhada de desempenho.

## 🛠️ Tecnologias Utilizadas
- Python 3.10+
- Scikit-learn
- Pandas
- Matplotlib/Seaborn
- Joblib
- Virtualenv (ambiente virtual)

## 📂 Estrutura de Arquivos
```
.
├── data/
│   └── Water_Quality_Prediction_red.csv       # Dataset principal
├── models/                                    # Modelos treinados
├── test_data/                                 # Dados de teste
├── src/
│   ├── main.py                # Ponto de entrada do sistema
│   ├── modelo_ml.py           # Lógica de treinamento e predição
│   └── avaliacao.py           # Avaliação de desempenho do modelo
├── requirements.txt           # Dependências do projeto
└── README.md
```

## 🔧 Pré-requisitos
- Python 3.10 ou superior
- Gerenciador de pacotes pip

## 🚀 Configuração do Ambiente

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_REPOSITORIO]
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## ▶️ Como Executar

1. Coloque o arquivo `Water_Quality_Prediction_red.csv` na pasta `data/`

2. Execute o programa principal:
```bash
python src/main.py
```

## 🧩 Funcionalidades

### Menu Principal
1. **Executar modelo de classificação**: Treina o modelo e permite fazer predições
2. **Exibir métricas de análise**: Mostra avaliações de desempenho (após treinamento)
3. **Sair**: Encerra o programa

### Modo de Predição
O sistema solicita 3 parâmetros principais:
- pH (0-14)
- Turbidez (0-1000)
- Condutividade (0-2000)

E classifica a água em três níveis de risco:
1. **Baixo risco** (≤33% contaminação)
2. **Risco moderado** (33-50%)
3. **Alto risco** (>50%)

### Avaliação de Desempenho
Após o treinamento, oferece:
- Relatório de classificação detalhado
- Métricas principais (Precisão, Recall, F1-Score)
- Matriz de confusão visual
- Curva ROC com cálculo de AUC

## ⚙️ Configuração Técnica
- **Algoritmo**: Random Forest Classifier
- **Pré-processamento**:
  - Imputação de valores faltantes (média)
  - Validação de faixas de entrada
- **Persistência**:
  - Modelos salvos em formato PKL
  - Dados de teste armazenados em CSV
- **Monitoramento**: Logs detalhados em arquivos

## 💾 Dados
O sistema espera encontrar o dataset na pasta `data/` com o nome:
`Water_Quality_Prediction_red.csv`

## 📊 Saídas Geradas
- Modelos treinados (`models/trained_model.pkl`)
- Dados de teste (`test_data/X_test.csv`, `test_data/y_test.csv`)
- Logs de execução (`*.log`)
- Gráficos interativos (matriz de confusão, curva ROC)

## 🔄 Fluxo de Trabalho
1. Treinamento do modelo com dados históricos
2. Validação com conjunto de teste (20% dos dados)
3. Persistência do modelo e dados de teste
4. Predição com inputs do usuário
5. Análise de desempenho com métricas diversas

## ⚠️ Notas Importantes
- O treinamento do modelo é necessário antes da avaliação de desempenho
- Apenas 3 parâmetros são coletados do usuário (demais são definidos automaticamente)
- Os diretórios `models/` e `test_data/` são criados automaticamente

Este projeto foi desenvolvido utilizando boas práticas de engenharia de machine learning, incluindo validação rigorosa de entradas, tratamento de erros e modularização de componentes.
