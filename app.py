from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import warnings

# Configurações
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# DataAnalysisAgent integrado no app.py
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

    def _initialize_agent(self):
        """Inicializa o agente com configuração correta."""
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        try:
            # Criar agente com allow_dangerous_code
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="openai-tools",
                allow_dangerous_code=True,  # OBRIGATÓRIO para pandas agent
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60
            )
            print("Agente pandas inicializado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar agente: {e}")
            # Fallback básico
            try:
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                print("Agente inicializado em modo básico")
            except Exception as e2:
                print(f"Falha na inicialização: {e2}")
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
                       f"Consultas disponíveis:\n"
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
            return min(int(numbers[0]), 50)
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

# Instância global do agente
agent = DataAnalysisAgent()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analisador de Dados IA</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #2c3e50; margin-bottom: 10px; }
            .header p { color: #7f8c8d; }
            .section { background: white; padding: 25px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #3498db; padding: 30px; text-align: center; border-radius: 8px; background: #ecf0f1; }
            .upload-area.dragover { background: #d5dbdb; border-color: #2980b9; }
            .chat-container { display: flex; flex-direction: column; height: 500px; }
            .chat-messages { flex: 1; overflow-y: auto; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
            .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
            .user-message { background: #3498db; color: white; margin-left: 20%; }
            .agent-message { background: #2ecc71; color: white; margin-right: 20%; }
            .system-message { background: #95a5a6; color: white; text-align: center; }
            .chat-input { display: flex; margin-top: 10px; }
            .chat-input input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
            .chat-input button { padding: 12px 20px; background: #3498db; color: white; border: none; border-radius: 6px; margin-left: 10px; cursor: pointer; }
            .chat-input button:hover { background: #2980b9; }
            .btn { padding: 12px 24px; background: #2ecc71; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
            .btn:hover { background: #27ae60; }
            .status { padding: 10px; border-radius: 6px; margin: 10px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
            .loading { display: none; text-align: center; padding: 20px; }
            .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Analisador de Dados com IA</h1>
                <p>Carregue seu CSV e faça perguntas inteligentes sobre os dados</p>
            </div>
            
            <div class="section">
                <h3>📁 Upload do Dataset</h3>
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="fileInput" accept=".csv" style="display: none;">
                    <p>Clique aqui ou arraste um arquivo CSV</p>
                    <button class="btn" onclick="document.getElementById('fileInput').click()">Selecionar Arquivo</button>
                </div>
                <div id="uploadStatus"></div>
            </div>
            
            <div class="section">
                <h3>💬 Chat com IA</h3>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message system-message">Carregue um arquivo CSV para começar a conversar!</div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="queryInput" placeholder="Ex: Quais são as primeiras 5 linhas? Qual a média da coluna Amount?" disabled>
                        <button onclick="sendQuery()" id="sendBtn" disabled>Enviar</button>
                    </div>
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <span>Processando...</span>
                </div>
            </div>
        </div>

        <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const chatMessages = document.getElementById('chatMessages');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');
        const uploadStatus = document.getElementById('uploadStatus');
        const loading = document.getElementById('loading');

        // Upload de arquivo
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

        async function handleFile(file) {
            if (!file || !file.name.endsWith('.csv')) {
                showStatus('Apenas arquivos .csv são aceitos!', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            
            try {
                showLoading(true);
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (result.success) {
                    showStatus(result.message, 'success');
                    enableChat();
                    addMessage('Sistema', result.message);
                } else {
                    showStatus(result.message, 'error');
                }
            } catch (error) {
                showStatus('Erro ao carregar arquivo: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function sendQuery() {
            const query = queryInput.value.trim();
            if (!query) return;
            
            addMessage('Você', query);
            queryInput.value = '';
            showLoading(true);
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const result = await response.json();
                addMessage('IA', result.response);
            } catch (error) {
                addMessage('Erro', 'Erro na consulta: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function addMessage(sender, message) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender === 'Você' ? 'user-message' : sender === 'IA' ? 'agent-message' : 'system-message'}`;
            
            if (sender === 'IA' || sender === 'Sistema') {
                messageDiv.innerHTML = `<strong>${sender}:</strong><pre>${message}</pre>`;
            } else {
                messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function showStatus(message, type) {
            uploadStatus.innerHTML = `<div class="status ${type}">${message}</div>`;
            setTimeout(() => uploadStatus.innerHTML = '', 5000);
        }

        function showLoading(show) {
            loading.style.display = show ? 'block' : 'none';
            sendBtn.disabled = show;
        }

        function enableChat() {
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.placeholder = 'Faça uma pergunta sobre seus dados...';
        }

        // Enter para enviar
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !sendBtn.disabled) sendQuery();
        });
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'Nenhum arquivo enviado', 'success': False}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'Arquivo inválido', 'success': False}), 400

        if file and file.filename.endswith('.csv'):
            # Salvar temporariamente
            file_path = f"/tmp/{file.filename}"
            file.save(file_path)
            
            # Carregar no agente
            if agent.load_csv(file_path):
                # Remover arquivo temporário
                try:
                    os.remove(file_path)
                except:
                    pass
                
                return jsonify({
                    'message': f'CSV carregado com sucesso! {len(agent.df)} linhas e {len(agent.df.columns)} colunas.',
                    'success': True
                })
            else:
                return jsonify({'message': 'Erro ao processar o arquivo CSV', 'success': False}), 500
        
        return jsonify({'message': 'Apenas arquivos .csv são aceitos', 'success': False}), 400
    
    except Exception as e:
        print(f"Erro no upload: {e}")
        return jsonify({'message': f'Erro interno: {str(e)}', 'success': False}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'Por favor, faça uma pergunta.'})
        
        # Usar o agente para processar a consulta
        response = agent.run_query(query)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Erro na consulta: {e}")
        return jsonify({'response': f'Erro ao processar a consulta: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Aplicação funcionando normalmente',
        'agent_status': 'loaded' if agent.df is not None else 'no_data'
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
