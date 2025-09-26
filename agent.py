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
        
        # Configuração do LLM SEM parâmetros problemáticos
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
            
            # Inicializar o agente com o parâmetro correto
            self._initialize_agent()
            return True
            
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return False

    def _initialize_agent(self):
        """Inicializa o agente com configuração correta."""
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        try:
            # SOLUÇÃO: allow_dangerous_code deve estar no create_pandas_dataframe_agent
            # MAS sem estar nas configurações do ChatOpenAI
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="openai-tools",
                allow_dangerous_code=True,  # AQUI é onde deve estar!
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60
            )
            print("✅ Agente pandas inicializado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar agente: {e}")
            # Se falhar, tenta com configuração mínima
            try:
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    allow_dangerous_code=True,  # Obrigatório aqui também
                    verbose=True
                )
                print("⚠️ Agente inicializado com configuração mínima")
            except Exception as e2:
                print(f"❌ Falha total na inicialização: {e2}")
                raise

    def run_query(self, query: str) -> str:
        """Executa uma consulta no agente."""
        if self.agent_executor is None:
            return "❌ Agente não inicializado. Por favor, carregue um CSV primeiro."
        
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
            print(f"❌ {error_msg}")
            
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
                    return "✅ Não há valores nulos no dataset!"
                else:
                    return f"Valores nulos por coluna:\n{null_info.to_string()}"
            
            elif any(word in query_lower for word in ['tipos', 'types', 'dtypes']):
                return f"Tipos de dados:\n{self.df.dtypes.to_string()}"
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas descritivas:\n{self.df.describe(include='all').to_string()}"
            
            else:
                return (f"❌ Não foi possível processar a consulta avançada devido a limitações técnicas.\n\n"
                       f"Consultas básicas disponíveis:\n"
                       f"- 'primeiras 10 linhas'\n"
                       f"- 'colunas do dataset'\n"
                       f"- 'informações do dataset'\n"
                       f"- 'valores nulos'\n"
                       f"- 'estatísticas descritivas'\n"
                       f"- 'tipos de dados'")
                
        except Exception as e:
            return f"❌ Erro ao processar consulta básica: {e}"

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

    def create_simple_visualization(self, column_name: str, plot_type: str = "histogram") -> str:
        """Cria visualizações simples sem usar o agente."""
        if self.df is None:
            return "❌ Nenhum dataset carregado."
        
        if column_name not in self.df.columns:
            available_cols = ", ".join(self.df.columns[:10])
            return f"❌ Coluna '{column_name}' não encontrada. Disponíveis: {available_cols}..."
        
        try:
            plt.figure(figsize=(10, 6))
            
            if plot_type.lower() == "histogram":
                if pd.api.types.is_numeric_dtype(self.df[column_name]):
                    data = self.df[column_name].dropna()
                    plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'Histograma - {column_name}')
                    plt.xlabel(column_name)
                    plt.ylabel('Frequência')
                else:
                    # Para dados categóricos
                    value_counts = self.df[column_name].value_counts().head(20)
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.title(f'Distribuição - {column_name}')
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            
            elif plot_type.lower() == "boxplot":
                if pd.api.types.is_numeric_dtype(self.df[column_name]):
                    data = self.df[column_name].dropna()
                    plt.boxplot(data)
                    plt.title(f'Boxplot - {column_name}')
                    plt.ylabel(column_name)
                else:
                    return f"❌ Boxplot requer coluna numérica. '{column_name}' é categórica."
            
            plt.tight_layout()
            
            # Salvar como base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return f"✅ Gráfico {plot_type} criado para a coluna '{column_name}'"
            
        except Exception as e:
            plt.close()
            return f"❌ Erro ao criar visualização: {e}"

# Exemplo de uso e teste
if __name__ == "__main__":
    agent = DataAnalysisAgent()
    print("🚀 DataAnalysisAgent inicializado!")
    
    # Teste básico
    # if agent.load_csv("seu_arquivo.csv"):
    #     print("✅ CSV carregado com sucesso")
    #     print(agent.run_query("primeiras 3 linhas"))
    # else:
    #     print("❌ Falha ao carregar CSV")
