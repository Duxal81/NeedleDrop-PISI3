import pandas as pd
import os
import logging
import csv
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def converter_csv_para_parquet(arquivo_csv, arquivo_parquet):
    """
    Converte um arquivo CSV para Parquet com tratamento de erros
    """
    try:
        logging.info(f"Convertendo: {arquivo_csv}")
        
        # LEITURA DO CSV COM TRATAMENTO DE ERROS
        df = pd.read_csv(
            arquivo_csv,
            encoding='utf-8',
            on_bad_lines='skip',      # Ignora linhas problemáticas
            quoting=csv.QUOTE_ALL,    # Força aspas em todos os campos
            engine='python',          # Engine mais tolerante a erros
            skipinitialspace=True,    # Ignora espaços após separador
            encoding_errors='ignore'  # Ignora erros de encoding
        )
        
        # Verifica se o DataFrame não está vazio
        if df.empty:
            logging.error(f"🍀 Arquivo CSV vazio ou todas as linhas foram ignoradas: {arquivo_csv}")
            return False
        
        # Cria diretório de destino se não existir
        os.makedirs(os.path.dirname(arquivo_parquet) if os.path.dirname(arquivo_parquet) else '.', exist_ok=True)
        
        # Salva como Parquet
        df.to_parquet(arquivo_parquet, index=False)
        logging.info(f"🍀 Conversão bem-sucedida: {arquivo_parquet}")
        return True
        
    except pd.errors.EmptyDataError:
        logging.error(f"🍀 Arquivo CSV vazio: {arquivo_csv}")
        return False
    except FileNotFoundError:
        logging.error(f"🍀 Arquivo não encontrado: {arquivo_csv}")
        return False
    except Exception as e:
        logging.error(f"🍀 Erro ao converter {arquivo_csv}: {e}")
        return False

def processar_conversao(caminhos_csv, pasta_saida='data_parquet'):
    """
    Processa a conversão de múltiplos arquivos CSV para Parquet
    """
    logging.info(f"🍀 Iniciando conversão de {len(caminhos_csv)} arquivos CSV para Parquet...")
    logging.info("---")
    
    sucessos = 0
    falhas = 0
    
    for arquivo_csv in caminhos_csv:
        if not os.path.exists(arquivo_csv):
            logging.error(f"🍀 Arquivo não encontrado: {arquivo_csv}")
            falhas += 1
            continue
            
        # Define o nome do arquivo de saída
        nome_base = os.path.splitext(os.path.basename(arquivo_csv))[0]
        arquivo_parquet = os.path.join(pasta_saida, f"{nome_base}.parquet")
        
        # Realiza a conversão
        if converter_csv_para_parquet(arquivo_csv, arquivo_parquet):
            sucessos += 1
        else:
            falhas += 1
        
        logging.info("---")
    
    # Relatório final
    logging.info("🍀 RELATÓRIO FINAL DA CONVERSÃO:")
    logging.info(f"🍀 Total de arquivos processados: {len(caminhos_csv)}")
    logging.info(f"🍀 Conversões bem-sucedidas: {sucessos}")
    logging.info(f"🍀 Conversões falhas: {falhas}")
    logging.info("🍀 Processo de conversão finalizado!")
    
    return sucessos, falhas

def main():
    """
    Função principal
    """
    # Lista de arquivos CSV para converter
    arquivos_csv = [
        'data/Spotify_Youtube.csv'
        # Adicione mais arquivos aqui se necessário
    ]
    
    # Processa a conversão
    sucessos, falhas = processar_conversao(arquivos_csv)
    
    # Encerra com código de erro se houver falhas
    exit(1 if falhas > 0 else 0)

if __name__ == "__main__":
    main()