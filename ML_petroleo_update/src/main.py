import os
import sys
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='w'
)

# Adiciona o diretório src ao path para importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_exists():
    """Verifica se o modelo treinado existe"""
    # Obter caminho absoluto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path = os.path.join(root_dir, "models", "trained_model.pkl")
    return os.path.exists(model_path)

def main():
    import avaliacao
    import modelo_ml

    # Obter caminhos para exibir a estrutura esperada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    while True:
        # Verificar se o modelo existe
        model_exists = check_model_exists()
        
        print("\n\033[1;33m=-=-=-= Menu Principal =-=-=-=\033[m")
        print("1. Executar modelo de classificação")
        
        if model_exists:
            print("2. Exibir métricas de análise de desempenho")
        else:
            print("\033[1;31m2. Exibir métricas (indisponível - treine o modelo primeiro)\033[m")
        
        print("3. Sair")
        
        try:
            choice = input("Escolha uma opção (1-3): ")
            choice = int(choice)
            print()
            
            if choice == 1:
                modelo_ml.executar_modelo()
            elif choice == 2:
                if model_exists:
                    avaliacao.menu()
                else:
                    print("Por favor, execute primeiro a opção 1 para treinar o modelo.")
            elif choice == 3:
                print("Saindo do programa...")
                break
            else:
                print("Opção inválida! Por favor, escolha 1, 2 ou 3.")
        except ValueError:
            print("Entrada inválida! Por favor, digite um número entre 1 e 3.")
        except Exception as e:
            logging.error(f"Erro inesperado: {e}", exc_info=True)
            print(f"Erro inesperado: {e}. Tente novamente.")

if __name__ == "__main__":
    main()