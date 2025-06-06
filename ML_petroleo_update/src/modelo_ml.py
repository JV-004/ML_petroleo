import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os
import re
import logging
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='modelo_ml.log',
    filemode='w'
)

# Obter caminhos absolutos
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
DATA_DIR = os.path.join(root_dir, "data")
MODEL_DIR = os.path.join(root_dir, "models")
TEST_DATA_DIR = os.path.join(root_dir, "test_data")
DATA_PATH = os.path.join(DATA_DIR, "Water_Quality_Prediction_red.csv")

# Valores mínimos e máximos para validação (apenas para os 3 parâmetros)
VALID_RANGES = {
    "pH": (0, 14),
    "Turbidity": (0, 1000),
    "Conductivity": (0, 2000)
}

# Lista de colunas na ordem correta
NUMERIC_COLS = [
    "pH", "Iron", "Nitrate", "Chloride", "Lead", "Zinc", "Turbidity", "Fluoride", "Copper", "Odor",
    "Sulfate", "Conductivity", "Chlorine", "Manganese", "Total Dissolved Solids",
    "Water Temperature", "Air Temperature", "Day"
]

def sanitize_input(value: str) -> float:
    """Sanitiza e converte input do usuário para float com segurança"""
    if re.match(r'^-?\d+(?:\.\d+)?$', value):
        return float(value)
    raise ValueError("Entrada inválida: use apenas números")

def validate_input(value: float, min_val: float, max_val: float) -> bool:
    """Valida se o valor está dentro da faixa permitida"""
    return min_val <= value <= max_val

def get_user_input() -> dict:
    """Coleta apenas pH, Turbidez e Condutividade do usuário"""
    user_data = {}
    print("\n--- Entrada de Dados ---")
    
    # Coletar apenas os 3 parâmetros principais
    for col in ["pH", "Turbidity", "Conductivity"]:
        min_val, max_val = VALID_RANGES[col]
        while True:
            try:
                prompt = f"{col} ({min_val}-{max_val}): "
                raw_input = input(prompt).strip()
                
                value = sanitize_input(raw_input)
                
                if validate_input(value, min_val, max_val):
                    user_data[col] = value
                    break
                else:
                    print(f"Valor fora da faixa permitida! ({min_val}-{max_val})")
            except ValueError as ve:
                print(f"Erro: {ve}")
            except Exception as e:
                logging.error(f"Erro na coleta de {col}: {e}")
                print(f"Erro inesperado. Tente novamente.")
    
    # Preencher as outras variáveis com valores fixos
    user_data["Iron"] = 0.1
    user_data["Nitrate"] = 10
    user_data["Chloride"] = 250
    user_data["Lead"] = 0.01
    user_data["Zinc"] = 5
    user_data["Fluoride"] = 0.7
    user_data["Copper"] = 1
    user_data["Odor"] = 0
    user_data["Sulfate"] = 250
    user_data["Chlorine"] = 0.2
    user_data["Manganese"] = 0.05
    user_data["Total Dissolved Solids"] = 500
    user_data["Water Temperature"] = 20
    user_data["Air Temperature"] = 25
    user_data["Day"] = 4
    
    return user_data

def executar_modelo():
    """Executa o pipeline completo de treinamento e predição"""
    try:
        # 1. VERIFICAR SE O ARQUIVO DE DADOS EXISTE
        if not os.path.exists(DATA_PATH):
            print(f"\n\033[1;31mERRO: Arquivo de dados não encontrado!\033[m")
            print(f"Por favor, coloque o arquivo 'Water_Quality_Prediction_red.csv' em:")
            print(f"\033[1;34m{DATA_DIR}\033[m")
            print("Crie o diretório 'data' se ele não existir.")
            return
        
        # 2. LEITURA DO DATASET
        print(f"Carregando dados de: \033[1;34m{DATA_PATH}\033[m")
        df = pd.read_csv(DATA_PATH)

        # 3. DEFINIR FEATURES E TARGET
        X = df[NUMERIC_COLS]
        y = df["Target"]

        # 4. DIVISÃO EM TREINO E TESTE (ANTES da imputação)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # 5. TRATAR VALORES AUSENTES (apenas com dados de treino)
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # 6. TREINAMENTO
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_imp, y_train)

        # 7. SALVAR O MODELO E OS DADOS
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
        imputer_path = os.path.join(MODEL_DIR, "imputer.pkl")
        X_test_path = os.path.join(TEST_DATA_DIR, "X_test.csv")
        y_test_path = os.path.join(TEST_DATA_DIR, "y_test.csv")
        
        joblib.dump(model, model_path)
        joblib.dump(imputer, imputer_path)
        
        pd.DataFrame(X_test_imp, columns=NUMERIC_COLS).to_csv(X_test_path, index=False)
        pd.Series(y_test).to_csv(y_test_path, index=False)

        print(f"\nModelo treinado salvo em: \033[1;34m{model_path}\033[m")
        print(f"Dados de teste salvos em: \033[1;34m{TEST_DATA_DIR}\033[m")

        # 8. PREDIÇÃO COM INPUT DO USUÁRIO
        while True:
            try:
                user_data = get_user_input()
                user_input = pd.DataFrame([user_data], columns=NUMERIC_COLS)
                
                # Carregar imputer e aplicar
                user_input_imp = pd.DataFrame(
                    imputer.transform(user_input), 
                    columns=user_input.columns
                )
                
                # Fazer predição
                probabilidade = model.predict_proba(user_input_imp)
                nivel_probabilidade = probabilidade[0][1] * 100

                print(f"\nNível de contaminação da água: {nivel_probabilidade:.2f}%")
                
                # Classificação de risco
                if nivel_probabilidade <= 33:
                    print('\nA água apresenta leve ou nenhuma contaminação por óleo.')
                    print('Evite contato direto com a água se possível.')
                    print('Observe se há manchas pequenas ou odor leve.')
                    print('Em caso de dúvidas, entre em contato com a defesa civil\n')
                elif nivel_probabilidade <= 50:
                    print('\033[1;33m\nCONTAMINAÇÃO MODERADA DETECTADA!\033[m')
                    print('Não entre em contato com a água.')
                    print('Notifique imediatamente as autoridades locais')
                    print('Evite o uso de sabão ou tentativas de limpeza caseira.')
                    print('Observe se animais estão sendo afetados')
                else:
                    print('\033[1;31mALTA CONTAMINAÇÃO DETECTADA\033[m')
                    print('Risco grave à saúde e ao meio ambiente')
                    print('Afaste-se da área contaminada')
                    print('Evacue se necessário e chame o órgão ambiental competente')
                    print('Registre o local para ajudar nas investigações')
                
                continuar = input('\nDeseja cadastrar novos dados? [S/N]: ').strip().upper()
                if continuar == 'N':
                    break
                    
            except Exception as e:
                logging.error(f"Erro na predição: {e}", exc_info=True)
                print(f"Erro durante a predição: {e}. Tente novamente.")
                
    except Exception as e:
        logging.error(f"Erro crítico em executar_modelo: {e}", exc_info=True)
        print(f"Erro crítico: {e}. O programa será encerrado.")