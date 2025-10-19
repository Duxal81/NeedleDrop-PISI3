import pandas as pd
import os
import logging
import csv
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def verificar_dependencias_parquet():
    try:
        import pyarrow
        return 'pyarrow'
    except ImportError:
        try:
            import fastparquet
            return 'fastparquet'
        except ImportError:
            logging.error("🍀 Nenhuma engine Parquet encontrada. Instale: pip install pyarrow")
            return None

def converter_csv_para_parquet(arquivo_csv, arquivo_parquet):
    
    try:
        logging.info(f"Convertendo: {arquivo_csv}")
        
        if not os.path.exists(arquivo_csv):
            logging.error(f"🍀 Arquivo não encontrado: {arquivo_csv}")
            return False
        
        df = pd.read_csv(
            arquivo_csv,
            encoding='utf-8',
            on_bad_lines='skip',      
            quoting=csv.QUOTE_ALL,  
            engine='python',         
            skipinitialspace=True,    
            encoding_errors='ignore'  
        )
        
        if df.empty:
            logging.error(f"🍀 Arquivo CSV vazio ou todas as linhas foram ignoradas: {arquivo_csv}")
            return False
        
        os.makedirs(os.path.dirname(arquivo_parquet) if os.path.dirname(arquivo_parquet) else '.', exist_ok=True)
        
        df.to_parquet(arquivo_parquet, index=False, engine='auto')
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

def processar_conversao(arquivos_csv, pasta_saida='data_parquet'):
    engine = verificar_dependencias_parquet()
    if not engine:
        return 0, len(arquivos_csv)
    
    logging.info(f"🍀 Iniciando conversão de {len(arquivos_csv)} arquivos CSV para Parquet...")
    logging.info(f"🍀 Usando engine: {engine}")
    logging.info("---")
    
    sucessos = 0
    falhas = 0
    
    for arquivo_csv in arquivos_csv:
  
        nome_base = os.path.splitext(os.path.basename(arquivo_csv))[0]
        arquivo_parquet = os.path.join(pasta_saida, f"{nome_base}.parquet")
        
        if converter_csv_para_parquet(arquivo_csv, arquivo_parquet):
            sucessos += 1
        else:
            falhas += 1
        
        logging.info("---")
    
    logging.info("🍀 RELATÓRIO FINAL DA CONVERSÃO:")
    logging.info(f"🍀 Total de arquivos processados: {len(arquivos_csv)}")
    logging.info(f"🍀 Conversões bem-sucedidas: {sucessos}")
    logging.info(f"🍀 Conversões falhas: {falhas}")
    logging.info("🍀 Processo de conversão finalizado!")
    
    return sucessos, falhas

def main():
    """
    Função principal
    """
    arquivos_csv = [
        'data/Spotify_Youtube.csv'
    ]
   

    sucessos, falhas = processar_conversao(arquivos_csv)
    
    exit(1 if falhas > 0 else 0)

if __name__ == "__main__":
    main()