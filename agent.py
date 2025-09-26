import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

class DataAnalysisAgent:
    def __init__(self):
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Configuração do LLM
        self.llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.agent_executor = None

    def load_csv(self, file_path):
        """Carrega um arquivo CSV e inicializa o agente."""
        try:
            self.df = pd.read_csv(file_path)
            print(f"CSV carregado com sucesso. {len(self.df)} linhas e {len(self.df.columns)} colunas.")
            
            # Inicializar o agente
            self._initialize_agent()
            return True
            
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return False

    def _create_visualization_tool(self):
        """Cria a ferramenta de visualização."""
        def generate_plot(query: str) -> str:
            """Gera gráficos baseados na query do usuário."""
            try:
                plt.style.use('default')
                query_lower = query.lower()
                
                # Parse da query para identificar tipo de gráfico e colunas
                if 'histograma' in query_lower or 'histogram' in query_lower:
                    return self._create_histogram(query)
                elif 'dispersão' in query_lower or 'scatter' in query_lower:
                    return self._create_scatter(query)
                elif 'barras' in query_lower or 'bar' in query_lower:
                    return self._create_barplot(query)
                elif 'linha' in query_lower or 'line' in query_lower:
                    return self._create_lineplot(query)
                else:
                    return ("Tipo de gráfico não reconhecido. Tente:\n"
                           "- 'histograma da coluna [nome]'\n"
                           "- 'dispersão entre [col1] e [col2]'\n"
                           "- 'gráfico de barras da coluna [nome]'\n"
                           "- 'gráfico de linha da coluna [nome]'")
                           
            except Exception as e:
                plt.close('all')
                return f"Erro ao gerar gráfico: {str(e)}"

        return Tool(
            name="generate_plot",
            func=generate_plot,
            description="Gera visualizações dos dados. Suporta histogramas, scatter plots, gráficos de barras e linha."
        )

    def _find_column_in_query(self, query: str):
        """Encontra o nome da coluna mencionada na query."""
        query_lower = query.lower()
        
        # Procura por padrões comuns
        for col in self.df.columns:
            if col.lower() in query_lower:
                return col
        
        # Procura por padrões como "coluna X"
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in ['coluna', 'da', 'de', 'column'] and i + 1 < len(words):
                potential_col = words[i + 1]
                for col in self.df.columns:
                    if col.lower() == potential_col:
                        return col
        
        return None

    def _create_histogram(self, query: str):
        """Cria um histograma."""
        column = self._find_column_in_query(query)
        
        if not column:
            return f"Coluna não encontrada. Colunas disponíveis: {', '.join(self.df.columns[:10])}"
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não existe no dataset."
        
        plt.figure(figsize=(10, 6))
        
        if pd.api.types.is_numeric_dtype(self.df[column]):
            data = self.df[column].dropna()
            plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Histograma de {column}')
            plt.xlabel(column)
            plt.ylabel('Frequência')
        else:
            # Para dados categóricos
            value_counts = self.df[column].value_counts().head(15)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'Distribuição de {column}')
            plt.xlabel(column)
            plt.ylabel('Contagem')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        return self._save_plot(f"Histograma gerado para a coluna '{column}'")

    def _create_scatter(self, query: str):
        """Cria um gráfico de dispersão."""
        # Encontra duas colunas na query
        mentioned_cols = []
        for col in self.df.columns:
            if col.lower() in query.lower():
                mentioned_cols.append(col)
        
        if len(mentioned_cols) < 2:
            return "Para scatter plot, mencione duas colunas numéricas."
        
        col1, col2 = mentioned_cols[0], mentioned_cols[1]
        
        if not (pd.api.types.is_numeric_dtype(self.df[col1]) and 
                pd.api.types.is_numeric_dtype(self.df[col2])):
            return f"Ambas as colunas ({col1}, {col2}) precisam ser numéricas."
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[col1], self.df[col2], alpha=0.6)
        plt.title(f'Dispersão: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True, alpha=0.3)
        
        return self._save_plot(f"Gráfico de dispersão criado para {col1} vs {col2}")

    def _create_barplot(self, query: str):
        """Cria um gráfico de barras."""
        column = self._find_column_in_query(query)
        
        if not column:
            return "Coluna não encontrada para gráfico de barras."
        
        plt.figure(figsize=(12, 6))
        
        if pd.api.types.is_numeric_dtype(self.df[column]):
            # Para dados numéricos, criar bins
            data = self.df[column].dropna()
            plt.hist(data, bins=20, alpha=0.7)
        else:
            # Para dados categóricos
            value_counts = self.df[column].value_counts().head(15)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        plt.title(f'Gráfico de Barras - {column}')
        plt.xlabel(column)
        plt.ylabel('Contagem')
        
        return self._save_plot(f"Gráfico de barras criado para {column}")

    def _create_lineplot(self, query: str):
        """Cria um gráfico de linha."""
        column = self._find_column_in_query(query)
        
        if not column or not pd.api.types.is_numeric_dtype(self.df[column]):
            return "Coluna numérica não encontrada para gráfico de linha."
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.df[column].dropna())
        plt.title(f'Gráfico de Linha - {column}')
        plt.xlabel('Índice')
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        
        return self._save_plot(f"Gráfico de linha criado para {column}")

    def _save_plot(self, message: str):
        """Salva o plot atual e retorna mensagem de confirmação."""
        try:
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return message
        except Exception as e:
            plt.close()
            return f"Erro ao salvar gráfico: {e}"

    def _initialize_agent(self):
        """Inicializa o agente com configuração correta."""
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        try:
            # Criar ferramenta de visualização
            plot_tool = self._create_visualization_tool()
            
            # Criar agente com allow_dangerous_code
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="openai-tools",
                allow_dangerous_code=True,  # OBRIGATÓRIO para pandas agent
                extra_tools=[plot_tool],
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60
            )
            print("Agente pandas inicializado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar agente com extra_tools: {e}")
            # Fallback sem extra_tools
            try:
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    verbose=True,
                    agent_type="openai-tools",
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    max_iterations=10
                )
                print("Agente inicializado em modo básico (sem visualizações)")
            except Exception as e2:
                print(f"Falha total na inicialização: {e2}")
                raise

    def run_query(self, query: str) -> str:
        """Executa uma consulta no agente."""
        if self.agent_executor is None:
            return "Agente não inicializado. Por favor, carregue um CSV primeiro."
        
        try:
            # Adiciona contexto sobre o dataset
            enhanced_query = f"""
Dataset Info:
- Total de linhas: {len(self.df)}
- Total de colunas: {len(self.df.columns)}
- Colunas disponíveis: {', '.join(self.df.columns[:10])}{'...' if len(self.df.columns) > 10 else ''}
- Tipos de dados: {dict(self.df.dtypes.value_counts())}

Consulta do usuário: {query}

Por favor, responda de forma clara e concisa.
"""
            
            result = self.agent_executor.invoke({"input": enhanced_query})
            return result.get("output", "Sem resposta disponível.")
            
        except Exception as e:
            error_msg = f"Erro ao processar consulta: {str(e)}"
            print(f"Erro: {error_msg}")
            
            # Fallback para consultas básicas
            return self._handle_simple_queries(query)

    def _handle_simple_queries(self, query: str) -> str:
        """Manipula consultas simples quando o agente principal falha."""
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
                    return "Não há valores nulos no dataset!"
                else:
                    return f"Valores nulos por coluna:\n{null_info.to_string()}"
            
            elif any(word in query_lower for word in ['tipos', 'types', 'dtypes']):
                return f"Tipos de dados:\n{self.df.dtypes.to_string()}"
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas descritivas:\n{self.df.describe(include='all').to_string()}"
            
            else:
                return (f"Consulta não reconhecida pelo sistema simplificado.\n\n"
                       f"Consultas básicas disponíveis:\n"
                       f"- 'primeiras 10 linhas'\n"
                       f"- 'colunas do dataset'\n"
                       f"- 'informações do dataset'\n"
                       f"- 'valores nulos'\n"
                       f"- 'estatísticas descritivas'\n"
                       f"- 'tipos de dados'")
                
        except Exception as e:
            return f"Erro ao processar consulta básica: {e}"

    def _extract_number(self, text: str, default: int = 5) -> int:
        """Extrai um número do texto, retorna default se não encontrar."""
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return min(int(numbers[0]), 50)  # Limita a 50 para performance
        return default

    def get_dataframe_info(self) -> dict:
        """Retorna informações básicas do DataFrame."""
        if self.df is None:
            return {"error": "Nenhum CSV carregado"}
        
        try:
            return {
                "status": "loaded",
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "column_names": list(self.df.columns),
                "dtypes": {str(k): str(v) for k, v in self.df.dtypes.items()},
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024**2), 2),
                "null_counts": self.df.isnull().sum().to_dict(),
                "numeric_columns": list(self.df.select_dtypes(include=['number']).columns),
                "categorical_columns": list(self.df.select_dtypes(include=['object']).columns)
            }
        except Exception as e:
            return {"error": f"Erro ao obter informações: {e}"}

# Exemplo de uso
if __name__ == "__main__":
    agent = DataAnalysisAgent()
    print("DataAnalysisAgent inicializado!")
