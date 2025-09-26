import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

class DataAnalysisAgent:
    def __init__(self):
        self.df = None

    def load_csv(self, file_path):
        """Carrega um arquivo CSV."""
        try:
            self.df = pd.read_csv(file_path)
            print(f"CSV carregado com sucesso. {len(self.df)} linhas e {len(self.df.columns)} colunas.")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return False

    def run_query(self, query: str) -> str:
        """Executa consultas básicas nos dados."""
        if self.df is None:
            return "❌ Nenhum dataset carregado. Por favor, carregue um CSV primeiro."
        
        query_lower = query.lower()
        
        try:
            if any(word in query_lower for word in ['primeiras', 'first', 'head', 'início']):
                n = self._extract_number(query, default=5)
                return f"Primeiras {n} linhas do dataset:\n\n{self.df.head(n).to_string()}"
            
            elif any(word in query_lower for word in ['últimas', 'last', 'tail', 'final']):
                n = self._extract_number(query, default=5)
                return f"Últimas {n} linhas do dataset:\n\n{self.df.tail(n).to_string()}"
            
            elif any(word in query_lower for word in ['colunas', 'columns']):
                cols_info = []
                for col in self.df.columns:
                    dtype = str(self.df[col].dtype)
                    nulls = self.df[col].isnull().sum()
                    cols_info.append(f"- {col}: {dtype} ({nulls} valores nulos)")
                return f"Colunas do dataset ({len(self.df.columns)} total):\n" + "\n".join(cols_info)
            
            elif any(word in query_lower for word in ['shape', 'tamanho', 'dimensões']):
                return f"Dimensões do dataset: {self.df.shape[0]} linhas × {self.df.shape[1]} colunas"
            
            elif any(word in query_lower for word in ['info', 'informações', 'resumo']):
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                info_str = buffer.getvalue()
                
                # Adiciona estatísticas básicas
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    stats = f"\n\nEstatísticas das colunas numéricas:\n{self.df[numeric_cols].describe()}"
                    info_str += stats
                
                return info_str
            
            elif any(word in query_lower for word in ['nulos', 'null', 'missing', 'vazios']):
                null_counts = self.df.isnull().sum()
                null_info = null_counts[null_counts > 0]
                if len(null_info) == 0:
                    return "✅ Não há valores nulos no dataset!"
                else:
                    return f"Valores nulos por coluna:\n{null_info.to_string()}"
            
            elif any(word in query_lower for word in ['tipos', 'types', 'dtypes']):
                return f"Tipos de dados:\n{self.df.dtypes.to_string()}"
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas descritivas:\n{self.df.describe(include='all').to_string()}"
            
            elif any(word in query_lower for word in ['média', 'mean', 'average']):
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    means = self.df[numeric_cols].mean()
                    return f"Médias das colunas numéricas:\n{means.to_string()}"
                else:
                    return "Não há colunas numéricas no dataset."
            
            elif any(word in query_lower for word in ['histograma', 'histogram']):
                return self._create_histogram(query)
            
            else:
                return (f"❌ Consulta não reconhecida. Consultas disponíveis:\n"
                       f"- 'primeiras 10 linhas'\n"
                       f"- 'colunas do dataset'\n"
                       f"- 'informações do dataset'\n"
                       f"- 'valores nulos'\n"
                       f"- 'estatísticas descritivas'\n"
                       f"- 'tipos de dados'\n"
                       f"- 'média das colunas'\n"
                       f"- 'histograma da coluna [nome]'")
                
        except Exception as e:
            return f"❌ Erro ao processar consulta: {e}"

    def _extract_number(self, text: str, default: int = 5) -> int:
        """Extrai um número do texto."""
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return min(int(numbers[0]), 50)
        return default

    def _find_column_in_query(self, query: str):
        """Encontra o nome da coluna na query."""
        query_lower = query.lower()
        
        for col in self.df.columns:
            if col.lower() in query_lower:
                return col
        
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in ['coluna', 'da', 'de', 'column'] and i + 1 < len(words):
                potential_col = words[i + 1]
                for col in self.df.columns:
                    if col.lower() == potential_col:
                        return col
        return None

    def _create_histogram(self, query: str):
        """Cria um histograma simples."""
        column = self._find_column_in_query(query)
        
        if not column:
            return f"Coluna não encontrada. Colunas disponíveis: {', '.join(self.df.columns[:10])}"
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não existe no dataset."
        
        try:
            plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(self.df[column]):
                data = self.df[column].dropna()
                plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Histograma de {column}')
                plt.xlabel(column)
                plt.ylabel('Frequência')
            else:
                value_counts = self.df[column].value_counts().head(15)
                plt.bar(range(len(value_counts)), value_counts.values)
                plt.title(f'Distribuição de {column}')
                plt.xlabel(column)
                plt.ylabel('Contagem')
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            
            plt.tight_layout()
            plt.close()
            
            return f"✅ Histograma criado para a coluna '{column}'"
            
        except Exception as e:
            plt.close()
            return f"❌ Erro ao criar histograma: {e}"

    def get_dataframe_info(self) -> dict:
        """Retorna informações básicas do DataFrame."""
        if self.df is None:
            return {"error": "Nenhum CSV carregado"}
