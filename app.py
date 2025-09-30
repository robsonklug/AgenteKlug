from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io
import base64
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import warnings
import sys
import traceback

# Configurações
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Log de inicialização
print("=" * 80)
print("INICIANDO ANALISADOR DE DADOS COM IA")
print("=" * 80)
print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}")

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"OpenAI API: Configurada ({api_key[:20]}...)")
else:
    print("OpenAI API: NAO CONFIGURADA - Configure em Environment Variables")
print("=" * 80 + "\n")

class DataAnalysisAgent:
    def __init__(self):
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = None
        self.error_message = None
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                self.llm = ChatOpenAI(
                    temperature=0, 
                    model="gpt-4o-mini",
                    openai_api_key=api_key
                )
                print("ChatOpenAI inicializado com sucesso")
            except Exception as e:
                self.llm = None
                self.error_message = f"Erro ao inicializar ChatOpenAI: {str(e)}"
                print(f"ERRO: {self.error_message}")
        else:
            self.llm = None
            self.error_message = "Chave da API OpenAI não configurada"

    def load_csv(self, file_path):
        print(f"\n{'='*80}")
        print(f"INICIANDO CARREGAMENTO DE CSV")
        print(f"{'='*80}")
        print(f"Arquivo: {os.path.basename(file_path)}")
        
        try:
            # Passo 1: Validar arquivo
            print("\n[1/4] Validando arquivo...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
            file_size = os.path.getsize(file_path)
            print(f"      Tamanho: {file_size / 1024:.2f} KB")
            
            # Passo 2: Carregar CSV
            print("\n[2/4] Carregando CSV...")
            self.df = pd.read_csv(file_path)
            print(f"      Linhas: {len(self.df):,}")
            print(f"      Colunas: {len(self.df.columns)}")
            print(f"      Colunas: {list(self.df.columns[:5])}{'...' if len(self.df.columns) > 5 else ''}")
            print(f"      Memória: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Passo 3: Verificar API
            print("\n[3/4] Verificando configuração da API...")
            if not os.environ.get("OPENAI_API_KEY"):
                print("      AVISO: API não configurada - apenas consultas básicas disponíveis")
                print("      Configure OPENAI_API_KEY no Render Dashboard > Environment")
                self.agent_executor = None
                return True
            
            if self.llm is None:
                print(f"      ERRO: {self.error_message}")
                return True
            
            print("      API configurada corretamente")
            
            # Passo 4: Inicializar agente
            print("\n[4/4] Inicializando agente de IA...")
            self._initialize_agent()
            
            print(f"\n{'='*80}")
            print("CSV CARREGADO COM SUCESSO")
            if self.agent_executor:
                print("Status: Agente IA ativo - Todas as funcionalidades disponíveis")
            else:
                print("Status: Modo básico - Apenas consultas simples disponíveis")
            print(f"{'='*80}\n")
            return True
            
        except pd.errors.EmptyDataError:
            self.error_message = "Arquivo CSV vazio ou inválido"
            print(f"\nERRO: {self.error_message}")
            return False
        except pd.errors.ParserError as e:
            self.error_message = f"Erro ao analisar CSV: {str(e)}"
            print(f"\nERRO: {self.error_message}")
            return False
        except UnicodeDecodeError:
            self.error_message = "Encoding inválido. Use UTF-8 ou tente outro formato"
            print(f"\nERRO: {self.error_message}")
            return False
        except MemoryError:
            self.error_message = "Arquivo muito grande. Limite: 50MB"
            print(f"\nERRO: {self.error_message}")
            return False
        except Exception as e:
            self.error_message = f"Erro inesperado: {str(e)}"
            print(f"\nERRO CRÍTICO:")
            print(f"  Tipo: {type(e).__name__}")
            print(f"  Mensagem: {str(e)}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            return False

    def _initialize_agent(self):
        if self.df is None or self.llm is None:
            return
        
        try:
            print("      Tentando inicialização completa com ferramentas...")
            
            # Criar ferramenta de visualização
            plot_tool = self._create_visualization_tool()
            
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="openai-tools",
                allow_dangerous_code=True,
                extra_tools=[plot_tool],
                handle_parsing_errors=True,
                max_iterations=15,
                max_execution_time=90
            )
            print("      Agente IA inicializado COM visualizações")
            
        except Exception as e:
            print(f"      Falha com ferramentas: {str(e)}")
            
            try:
                print("      Tentando inicialização básica...")
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    verbose=True,
                    agent_type="openai-tools",
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    max_iterations=10
                )
                print("      Agente IA inicializado em modo BÁSICO")
                
            except Exception as e2:
                print(f"      Falha total: {str(e2)}")
                print("      Continuando SEM agente IA")
                self.agent_executor = None

    def _create_visualization_tool(self):
        def generate_plot(query: str) -> str:
            try:
                query_lower = query.lower()
                
                if 'histograma' in query_lower or 'histogram' in query_lower:
                    return self._create_histogram(query)
                elif 'dispersão' in query_lower or 'scatter' in query_lower:
                    return self._create_scatter(query)
                else:
                    return "Tipo de gráfico não reconhecido. Use 'histograma da coluna X' ou 'dispersão entre X e Y'"
            except Exception as e:
                return f"Erro ao gerar gráfico: {str(e)}"

        return Tool(
            name="generate_plot",
            func=generate_plot,
            description="Gera visualizações dos dados (histogramas, scatter plots)"
        )

    def _find_column(self, query: str):
        query_lower = query.lower()
        for col in self.df.columns:
            if col.lower() in query_lower:
                return col
        return None

    def _create_histogram(self, query: str):
        column = self._find_column(query)
        if not column or column not in self.df.columns:
            return f"Coluna não encontrada. Disponíveis: {', '.join(self.df.columns[:10])}"
        
        try:
            plt.figure(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(self.df[column]):
                plt.hist(self.df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            else:
                value_counts = self.df[column].value_counts().head(15)
                plt.bar(range(len(value_counts)), value_counts.values)
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            
            plt.title(f'Distribuição de {column}')
            plt.tight_layout()
            plt.close()
            return f"Histograma gerado para '{column}'"
        except Exception as e:
            plt.close()
            return f"Erro ao criar histograma: {str(e)}"

    def _create_scatter(self, query: str):
        cols = [col for col in self.df.columns if col.lower() in query.lower()]
        if len(cols) < 2:
            return "Para scatter plot, mencione duas colunas numéricas"
        
        col1, col2 = cols[0], cols[1]
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[col1], self.df[col2], alpha=0.6)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f'{col1} vs {col2}')
            plt.tight_layout()
            plt.close()
            return f"Scatter plot criado: {col1} vs {col2}"
        except Exception as e:
            plt.close()
            return f"Erro ao criar scatter: {str(e)}"

    def run_query(self, query: str) -> str:
        print(f"\n{'='*80}")
        print(f"CONSULTA: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"{'='*80}")
        
        if self.df is None:
            error = "Nenhum dataset carregado. Faça upload de um arquivo CSV primeiro."
            print(f"ERRO: {error}")
            return error
        
        if self.agent_executor is None:
            print("Usando modo de consultas básicas (agente IA não disponível)")
            return self._handle_basic_queries(query)
        
        try:
            print("Processando com agente IA...")
            
            context = f"""
Dataset carregado:
- Linhas: {len(self.df):,}
- Colunas: {len(self.df.columns)}
- Nomes das colunas: {', '.join(self.df.columns[:12])}{'...' if len(self.df.columns) > 12 else ''}
- Tipos: {dict(self.df.dtypes.value_counts().items())}

Consulta do usuário: {query}

Responda de forma clara, direta e estruturada. Se fizer cálculos, mostre os resultados.
"""
            
            result = self.agent_executor.invoke({"input": context})
            response = result.get("output", "Sem resposta disponível")
            print("Resposta gerada com sucesso")
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"ERRO no agente IA ({error_type}): {str(e)}")
            
            if "rate limit" in str(e).lower():
                return "Limite de requisições atingido na API OpenAI. Aguarde alguns segundos e tente novamente."
            elif "invalid api key" in str(e).lower():
                return "Chave da API OpenAI inválida. Verifique a configuração no Render Dashboard."
            elif "insufficient_quota" in str(e).lower():
                return "Cota da API OpenAI esgotada. Adicione créditos em platform.openai.com/account/billing"
            else:
                print("Tentando modo básico...")
                return self._handle_basic_queries(query)

    def _handle_basic_queries(self, query: str) -> str:
        query_lower = query.lower()
        
        try:
            if any(word in query_lower for word in ['primeiras', 'first', 'head']):
                n = self._extract_number(query, 5)
                return f"Primeiras {n} linhas:\n\n{self.df.head(n).to_string()}"
            
            elif any(word in query_lower for word in ['últimas', 'last', 'tail']):
                n = self._extract_number(query, 5)
                return f"Últimas {n} linhas:\n\n{self.df.tail(n).to_string()}"
            
            elif any(word in query_lower for word in ['colunas', 'columns']):
                info = [f"- {col}: {self.df[col].dtype}, {self.df[col].isnull().sum()} nulos, {self.df[col].nunique()} únicos" 
                        for col in self.df.columns]
                return f"Colunas ({len(self.df.columns)} total):\n" + "\n".join(info)
            
            elif any(word in query_lower for word in ['tamanho', 'shape', 'dimensões']):
                return f"Dimensões: {self.df.shape[0]:,} linhas × {self.df.shape[1]} colunas"
            
            elif any(word in query_lower for word in ['info', 'informações', 'resumo']):
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                info = buffer.getvalue()
                numeric = self.df.select_dtypes(include=['number'])
                if len(numeric.columns) > 0:
                    info += f"\n\nEstatísticas numéricas:\n{numeric.describe()}"
                return info
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas descritivas:\n\n{self.df.describe(include='all').to_string()}"
            
            elif any(word in query_lower for word in ['nulos', 'null', 'missing']):
                nulls = self.df.isnull().sum()
                result = nulls[nulls > 0]
                if len(result) == 0:
                    return "Não há valores nulos no dataset"
                return f"Valores nulos ({nulls.sum()} total):\n\n{result.to_string()}"
            
            elif any(word in query_lower for word in ['duplicadas', 'duplicate']):
                dups = self.df.duplicated().sum()
                return f"Linhas duplicadas: {dups}" if dups > 0 else "Não há linhas duplicadas"
            
            elif 'correlação' in query_lower or 'correlation' in query_lower:
                numeric = self.df.select_dtypes(include=['number'])
                if len(numeric.columns) >= 2:
                    return f"Matriz de correlação:\n\n{numeric.corr().to_string()}"
                return "Precisa de pelo menos 2 colunas numéricas para correlação"
            
            else:
                return """Consultas básicas disponíveis:
- primeiras N linhas
- últimas N linhas  
- colunas / informações das colunas
- tamanho / dimensões
- informações completas / resumo
- estatísticas / describe
- valores nulos
- linhas duplicadas
- correlação (para colunas numéricas)

Para análises avançadas com IA, configure a chave OPENAI_API_KEY."""
                
        except Exception as e:
            return f"Erro ao processar consulta básica: {str(e)}"

    def _extract_number(self, text: str, default: int = 5) -> int:
        import re
        numbers = re.findall(r'\d+', text)
        return min(int(numbers[0]), 100) if numbers else default

agent = DataAnalysisAgent()

@app.route('/')
def index():
    api_status = "Configurada" if os.environ.get("OPENAI_API_KEY") else "Não configurada"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analisador de Dados com IA</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ 
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
            .api-status {{
                display: inline-block;
                padding: 8px 20px;
                background: {"rgba(76, 175, 80, 0.3)" if os.environ.get("OPENAI_API_KEY") else "rgba(244, 67, 54, 0.3)"};
                border-radius: 20px;
                margin-top: 10px;
                font-size: 0.9em;
            }}
            .content {{ padding: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .section-title {{ 
                color: #667eea;
                font-size: 1.5em;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }}
            .upload-area {{
                border: 3px dashed #667eea;
                padding: 50px;
                text-align: center;
                border-radius: 15px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                cursor: pointer;
                transition: all 0.3s;
            }}
            .upload-area:hover {{
                background: linear-gradient(135deg, #e8ecf1 0%, #b8c6db 100%);
                transform: translateY(-2px);
            }}
            .chat-container {{
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
            }}
            .chat-messages {{
                height: 500px;
                overflow-y: auto;
                padding: 25px;
                background: #fafafa;
            }}
            .message {{
                margin: 15px 0;
                padding: 15px 20px;
                border-radius: 15px;
                animation: slideIn 0.3s ease-out;
                max-width: 85%;
            }}
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .user {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
            }}
            .agent {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }}
            .system {{
                background: #e0e0e0;
                color: #333;
                text-align: center;
                margin: 10px auto;
            }}
            .error-msg {{
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                color: white;
            }}
            .chat-input {{
                display: flex;
                gap: 15px;
                padding: 20px;
                background: white;
                border-top: 2px solid #e0e0e0;
            }}
            .chat-input input {{
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 15px;
                transition: border-color 0.3s;
            }}
            .chat-input input:focus {{
                outline: none;
                border-color: #667eea;
            }}
            button {{
                padding: 15px 35px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                font-size: 15px;
                transition: all 0.3s;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }}
            button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }}
            .status {{
                padding: 15px 20px;
                border-radius: 10px;
                margin: 15px 0;
                font-weight: 500;
            }}
            .success {{ background: #d4edda; color: #155724; border-left: 4px solid #28a745; }}
            .error {{ background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }}
            pre {{
                background: rgba(0,0,0,0.05);
                padding: 12px;
                border-radius: 8px;
                font-size: 13px;
                overflow-x: auto;
                margin-top: 8px;
                white-space: pre-wrap;
            }}
            .loading {{
                display: none;
                text-align: center;
                padding: 20px;
            }}
            .spinner {{
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Analisador de Dados com IA</h1>
                <p>Powered by GPT-4o-mini + LangChain + Pandas</p>
                <div class="api-status">API OpenAI: {api_status}</div>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2 class="section-title">1. Upload do Dataset</h2>
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" accept=".csv" style="display: none;">
                        <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
                        <h3>Arraste um arquivo CSV aqui</h3>
                        <p style="color: #666; margin-top: 10px;">ou clique para selecionar</p>
                        <p style="color: #999; margin-top: 15px; font-size: 0.9em;">Tamanho máximo: 50MB</p>
                    </div>
                    <div id="uploadStatus"></div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">2. Análise com IA</h2>
                    <div class="chat-container">
                        <div class="chat-messages" id="chatMessages">
                            <div class="message system">
                                Carregue um arquivo CSV para começar a análise com inteligência artificial
                            </div>
                        </div>
                        <div class="chat-input">
                            <input type="text" id="queryInput" 
                                   placeholder="Ex: Mostre as primeiras linhas, Qual a média da coluna X?, Gere um histograma..." 
                                   disabled>
                            <button onclick="sendQuery()" id="sendBtn" disabled>Enviar</button>
                        </div>
                    </div>
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p style="margin-top: 15px; color: #667eea; font-weight: 500;">Processando com IA...</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
        const fileInput = document.getElementById('fileInput');
        const chatMessages = document.getElementById('chatMessages');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');

        fileInput.addEventListener('change', (e) => uploadFile(e.target.files[0]));

        async function uploadFile(file) {{
            if (!file) return;
            
            if (!file.name.endsWith('.csv')) {{
                showStatus('Apenas arquivos .csv são aceitos', 'error');
                return;
            }}

            const formData = new FormData();
            formData.append('file', file);
            
            showStatus('Carregando e processando arquivo...', 'success');
            showLoading(true);
            
            try {{
                const response = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                const result = await response.json();
                
                showStatus(result.message, result.success ? 'success' : 'error');
                
                if (result.success) {{
                    addMessage('Sistema', result.message);
                    queryInput.disabled = false;
                    sendBtn.disabled = false;
                    queryInput.focus();
                    queryInput.placeholder = "Faça uma pergunta sobre seus dados...";
                }}
            }} catch (error) {{
                showStatus('Erro de conexão: ' + error.message, 'error');
                addMessage('Erro', 'Falha ao carregar arquivo. Verifique sua conexão e tente novamente.');
            }} finally {{
                showLoading(false);
            }}
        }}

        async function sendQuery() {{
            const query = queryInput.value.trim();
            if (!query) return;
            
            addMessage('Você', query);
            queryInput.value = '';
            showLoading(true);
            
            try {{
                const response = await fetch('/ask', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query: query }})
                }});
                
                if (!response.ok) {{
                    throw new Error(`HTTP error! status: ${{response.status}}`);
                }}
                
                const result = await response.json();
                addMessage('IA', result.response);
            }} catch (error) {{
                addMessage('Erro', 'Falha na comunicação com o servidor: ' + error.message);
            }} finally {{
                showLoading(false);
                queryInput.focus();
            }}
        }}

        function addMessage(sender, message) {{
            const div = document.createElement('div');
            let className = 'message ';
            
            if (sender === 'Você') className += 'user';
            else if (sender === 'IA') className += 'agent';
            else if (sender === 'Erro') className += 'error-msg';
            else className += 'system';
            
            div.className = className;
            div.innerHTML = `<strong>${{sender}}:</strong><pre>${{message}}</pre>`;
            
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }}

        function showStatus(message, type) {{
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = `<div class="status ${{type}}">${{message}}</div>`;
            setTimeout(() => statusDiv.innerHTML = '', 8000);
        }}

        function showLoading(show) {{
            loading.style.display = show ? 'block' : 'none';
            sendBtn.disabled = show;
        }}

        queryInput.addEventListener('keypress', (e) => {{
            if (e.key === 'Enter' && !sendBtn.disabled) sendQuery();
        }});
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"\n{'='*80}")
    print("ENDPOINT /upload")
    print(f"{'='*80}")
    
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'Nenhum arquivo foi enviado na requisição', 'success': False}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'Nome de arquivo in
