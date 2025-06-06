import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Tuple, Any

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='avaliacao.log',
    filemode='w'
)

# Obter caminhos absolutos
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(root_dir, "models")
TEST_DATA_DIR = os.path.join(root_dir, "test_data")

# Variáveis globais para cache
_model = None
_X_test = None
_y_test = None
_y_pred = None

def init() -> None:
    """Inicializa os recursos apenas uma vez"""
    global _model, _X_test, _y_test, _y_pred
    if _model is None:
        try:
            model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
            X_test_path = os.path.join(TEST_DATA_DIR, "X_test.csv")
            y_test_path = os.path.join(TEST_DATA_DIR, "y_test.csv")
            
            _model = joblib.load(model_path)
            _X_test = pd.read_csv(X_test_path)
            _y_test = pd.read_csv(y_test_path).values.ravel()
            _y_pred = _model.predict(_X_test)
        except Exception as e:
            logging.error(f"Erro ao carregar recursos: {e}", exc_info=True)
            raise

def relatorio_classificacao() -> None:
    """Exibe relatório detalhado de classificação"""
    init()
    print("Relatório de Classificação Detalhado:")
    print(classification_report(_y_test, _y_pred, target_names=["Água Boa (0)", "Água Ruim (1)"]))

def precisao_recall_f1() -> None:
    """Calcula e exibe métricas principais"""
    init()
    precision = precision_score(_y_test, _y_pred)
    recall = recall_score(_y_test, _y_pred)
    f1 = f1_score(_y_test, _y_pred)
    print(f"\nPrecisão (Precision): {precision:.2f}")
    print(f"Recall (Sensibilidade): {recall:.2f}")
    print(f"F1-Score: {f1:.2f}\n")

def matriz_confusao() -> None:
    """Exibe matriz de confusão visual"""
    init()
    conf_matrix = confusion_matrix(_y_test, _y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["Água Boa (0)", "Água Ruim (1)"],
        yticklabels=["Água Boa (0)", "Água Ruim (1)"]
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    plt.show()

def curva_roc() -> None:
    """Plota curva ROC com AUC"""
    init()
    y_prob = _model.predict_proba(_X_test)[:, 1]
    fpr, tpr, _ = roc_curve(_y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()

def menu() -> None:
    """Menu interativo de avaliação"""
    while True:
        print('\n\033[1;33m---------- ÍNDICE ----------\033[m')
        print('1 - Relatório de classificação')
        print('2 - Precisão | Recall | F1')
        print('3 - Matriz de confusão')
        print('4 - Curva ROC')
        print('5 - Sair do menu de avaliações')

        try:
            indice = int(input('\n\033[1;33mDigite valor do índice: \033[m'))
            print()
            if indice == 1:
                relatorio_classificacao()
            elif indice == 2:
                precisao_recall_f1()
            elif indice == 3:
                matriz_confusao()
            elif indice == 4:
                curva_roc()
            elif indice == 5:
                break
            else:
                print('Entrada inválida! Por favor, digite um número entre 1 e 5.')
        except ValueError:
            print("Entrada inválida! Por favor, digite um número.")
        except Exception as e:
            logging.error(f"Erro no menu de avaliação: {e}", exc_info=True)
            print(f"Erro inesperado: {e}. Tente novamente.")