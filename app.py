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
from datetime import datetime
from io import StringIO

# Configurações
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')
load_dotenv()

# Lê a chave apenas uma vez
API_KEY = os.environ.get("OPENAI_API_KEY")

# Custom stream to capture print statements
class LogStream(StringIO):
    def __init__(self):
        super().__init__()
        self.logs = []

    def write(self, text):
        super().write(text)
        if isinstance(text, str) and text.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logs.append(f"[{timestamp}] {text.strip()}")
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]

    def flush(self):
        pass

# Redireciona stdout para o log interno
log_stream = LogStream()
sys.stdout = log_stream

# Mensagens iniciais
print("=" * 80)
print("INICIANDO ANALISADOR DE DADOS COM IA")
print("=" * 80)
print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}")

if API_KEY:
    print(f"✅ OPENAI_API_KEY detectada. Tamanho: {len(API_KEY)}")
    print(f"OpenAI API: Configurada ({API_KEY[:20]}...)")
else:
    print("❌ OPENAI_API_KEY não encontrada no ambiente.")
    print("OpenAI API: NAO CONFIGURADA")
print("=" * 80 + "\n")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

class DataAnalysisAgent:
    def __init__(self):
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = None
        self.error_message = None
        
        if API_KEY:
            try:
                self.llm = ChatOpenAI(
                    temperature=0, 
                    model="gpt-4o-mini",
                    openai_api_key=API_KEY
                )
                print("ChatOpenAI inicializado")
            except Exception as e:
                self.llm = None
                self.error_message = f"Erro ao inicializar ChatOpenAI: {str(e)}"
                print(f"ERRO: {self.error_message}")
        else:
            self.llm = None
            self.error_message = "Chave da API OpenAI não configurada"
            print(self.error_message)

    def load_csv(self, file_path):
        print(f"\n{'='*80}")
        print(f"CARREGANDO CSV")
        print(f"{'='*80}")
        
        try:
            print("[1/4] Validando arquivo...")
            if not os.path.exists(file_path):
                raise FileNotFoundError("Arquivo não encontrado")
            
            print("[2/4] Carregando CSV...")
            self.df = pd.read_csv(file_path)
            print(f"      Linhas: {len(self.df):,}")
            print(f"      Colunas: {len(self.df.columns)}")
            
            print("[3/4] Verificando API...")
            if not API_KEY:
                print("      AVISO: API não configurada")
                self.agent_executor = None
                return True
            
            if self.llm is None:
                print(f"      ERRO: {self.error_message}")
                return True
            
            print("[4/4] Inicializando agente IA...")
            self._initialize_agent()
            
            print(f"\n{'='*80}")
            print("CSV CARREGADO COM SUCESSO")
            print(f"{'='*80}\n")
            return True
            
        except Exception as e:
            self.error_message = f"Erro: {str(e)}"
            print(f"\nERRO: {self.error_message}")
            return False

    def _initialize_agent(self):
        if self.df is None or self.llm is None:
            return
        
        try:
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=False,
                agent_type="openai-tools",
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=20
            )
            print("      Agente IA inicializado")
            
        except Exception as e:
            print(f"      Erro: {str(e)}")
            self.agent_executor = None

    def run_query(self, query: str) -> dict:
        if self.df is None:
            return {"response": "Nenhum dataset carregado. Faça upload de um CSV primeiro.", "image": None}

        if self.agent_executor is None:
            return {"response": self._handle_basic_queries(query), "image": None}

        try:
            context = f"""
Dataset: {len(self.df):,} linhas, {len(self.df.columns)} colunas
Colunas: {', '.join(self.df.columns[:10])}

Pergunta: {query}
"""
            plt.clf()

            print(f"[run_query] Executando agente para pergunta: {query[:200]}")
            result = self.agent_executor.invoke({"input": context})
            response_text = result.get("output", "Sem resposta")

            image_data = None
            try:
                if plt.get_fignums():
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    plt.close()
            except Exception as e:
                print(f"⚠️ Erro ao salvar gráfico: {e}")
                image_data = None

            print(f"[run_query] Resposta gerada. image? {'sim' if image_data else 'não'}")
            return {"response": response_text, "image": image_data}

        except Exception as e:
            try:
                plt.close()
            except:
                pass

            msg = str(e).lower()
            print(f"[run_query] ERRO durante execução do agente: {e}")
            if "rate limit" in msg:
                return {"response": "⚠️ Limite de requisições atingido. Aguarde alguns segundos.", "image": None}
            elif "invalid api key" in msg:
                return {"response": "❌ Chave da API inválida. Verifique a configuração.", "image": None}
            elif "insufficient_quota" in msg:
                return {"response": "💸 Cota esgotada. Adicione créditos em platform.openai.com", "image": None}
            else:
                return {"response": self._handle_basic_queries(query), "image": None}

    def _handle_basic_queries(self, query: str) -> str:
        query_lower = query.lower()
        
        try:
            if 'primeiras' in query_lower or 'head' in query_lower:
                n = self._extract_number(query, 5)
                return f"Primeiras {n} linhas:\n{self.df.head(n).to_string()}"
            
            elif 'colunas' in query_lower or 'columns' in query_lower:
                info = [f"- {col}: {self.df[col].dtype}" for col in self.df.columns]
                return f"Colunas ({len(self.df.columns)}):\n" + "\n".join(info)
            
            elif 'tamanho' in query_lower or 'shape' in query_lower:
                return f"Dimensões: {self.df.shape[0]:,} linhas x {self.df.shape[1]} colunas"
            
            elif 'info' in query_lower:
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                return buffer.getvalue()
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas:\n{self.df.describe().to_string()}"
            
            elif 'nulos' in query_lower or 'null' in query_lower:
                nulls = self.df.isnull().sum()
                result = nulls[nulls > 0]
                return f"Valores nulos:\n{result.to_string()}" if len(result) > 0 else "Sem valores nulos"
            
            else:
                return (
                    "Consultas disponíveis:\n"
                    "- primeiras N linhas\n"
                    "- colunas\n"
                    "- tamanho\n"
                    "- info\n"
                    "- describe / estatísticas\n"
                    "- valores nulos"
                )
                
        except Exception as e:
            print(f"[handle_basic_queries] Erro: {e}")
            return f"Erro: {str(e)}"

    def _extract_number(self, text: str, default: int = 5) -> int:
        import re
        numbers = re.findall(r'\d+', text)
        return min(int(numbers[0]), 100) if numbers else default

agent = DataAnalysisAgent()

@app.route('/')
def index():
    api_status = "Configurada" if API_KEY else "Não configurada"
    status_color = "#4CAF50" if API_KEY else "#f44336"
    
    # IMPORTANTE: Usar """ para evitar conflitos com f-string
    # OU escapar TODAS as chaves do JavaScript com {{}}
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Analisador de Dados IA</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .api-status {
            display: inline-block;
            padding: 8px 20px;
            background: ''' + status_color + ''';
            border-radius: 20px;
            margin-top: 10px;
        }
        .content { padding: 40px; }
        .section { margin-bottom: 40px; }
        .section-title { 
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .upload-area {
            border: 3px dashed #667eea;
            padding: 50px;
            text-align: center;
            border-radius: 15px;
            background: #f5f7fa;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e8ecf1;
            transform: translateY(-2px);
        }
        .chat-container {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
        }
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 25px;
            background: #fafafa;
        }
        .log-container {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 20px;
        }
        .log-messages {
            max-height: 300px;
            overflow-y: auto;
            padding: 25px;
            background: #fafafa;
            font-family: monospace;
            font-size: 13px;
        }
        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 15px;
            animation: slideIn 0.3s;
            max-width: 85%;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
        }
        .agent {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .system {
            background: #e0e0e0;
            color: #333;
            text-align: center;
            margin: 10px auto;
        }
        .chat-input {
            display: flex;
            gap: 15px;
            padding: 20px;
            background: white;
            border-top: 2px solid #e0e0e0;
        }
        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 15px;
        }
        .chat-input input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 15px 35px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .status {
            padding: 15px 20px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        pre {
            background: rgba(0,0,0,0.05);
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .plot-image {
            max-width: 100%;
            margin-top: 10px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Analisador de Dados com IA</h1>
            <p>GPT-4o-mini + LangChain + Pandas</p>
            <div class="api-status">API: ''' + api_status + '''</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">1. Upload CSV</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept=".csv" style="display: none;">
                    <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
                    <h3>Arraste um arquivo CSV aqui</h3>
                    <p style="color: #666; margin-top: 10px;">ou clique para selecionar</p>
                </div>
                <div id="uploadStatus"></div>
            </div>
            
            <div class="section">
                <h2 class="section-title">2. Análise com IA</h2>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message system">Carregue um CSV para começar</div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="queryInput" placeholder="Faça uma pergunta..." disabled>
                        <button onclick="sendQuery()" id="sendBtn" disabled>Enviar</button>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">3. Logs do Sistema</h2>
                <div class="log-container">
                    <div class="log-messages" id="logMessages">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    const fileInput = document.getElementById('fileInput');
    const chatMessages = document.getElementById('chatMessages');
    const logMessages = document.getElementById('logMessages');
    const queryInput = document.getElementById('queryInput');
    const sendBtn = document.getElementById('sendBtn');

    fileInput.addEventListener('change', (e) => uploadFile(e.target.files[0]));

    async function uploadFile(file) {
        if (!file || !file.name.endsWith('.csv')) {
            showStatus('Apenas arquivos .csv', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        
        showStatus('Carregando...', 'success');
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            showStatus(result.message, result.success ? 'success' : 'error');
            
            if (result.success) {
                addMessage('Sistema', result.message);
                queryInput.disabled = false;
                sendBtn.disabled = false;
                queryInput.focus();
            }
        } catch (error) {
            showStatus('Erro: ' + error.message, 'error');
        }
    }

    async function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        addMessage('Você', query);
        queryInput.value = '';
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            const result = await response.json();
            console.log("Resposta do backend:", result);
            addMessage('IA', result.response, result.image);
        } catch (error) {
            addMessage('Erro', error.message);
        }
    }

    function addMessage(sender, message, image = null) {
        const div = document.createElement('div');
        let className = 'message ';
        if (sender === 'Você') className += 'user';
        else if (sender === 'IA') className += 'agent';
        else className += 'system';
        
        div.className = className;
        div.innerHTML = `<strong>${sender}:</strong><pre>${message}</pre>`;
        
        if (image) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${image}`;
            img.className = 'plot-image';
            div.appendChild(img);
        }
        
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showStatus(message, type) {
        const statusDiv = document.getElementById('uploadStatus');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        setTimeout(() => statusDiv.innerHTML = '', 8000);
    }

    async function fetchLogs() {
        try {
            const response = await fetch('/logs');
            const result = await response.json();
            logMessages.innerHTML = result.logs.map(log => `<pre>${log}</pre>`).join('');
            logMessages.scrollTop = logMessages.scrollHeight;
        } catch (error) {
            console.error('Erro ao carregar logs:', error);
        }
    }

    setInterval(fetchLogs, 2000);
    fetchLogs();

    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !sendBtn.disabled) sendQuery();
    });
    </script>
</body>
</html>
    '''
    return html

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'Nenhum arquivo enviado', 'success': False}), 400

        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'message': 'Arquivo inválido. Use apenas .csv', 'success': False}), 400

        file_path = f"/tmp/{file.filename}"
        file.save(file_path)
        
        if agent.load_csv(file_path):
            try:
                os.remove(file_path)
            except:
                pass
            
            if agent.agent_executor:
                msg = f"CSV carregado! {len(agent.df):,} linhas x {len(agent.df.columns)} colunas. Agente IA ativo."
            else:
                msg = f"CSV carregado: {len(agent.df):,} linhas x {len(agent.df.columns)} colunas. Modo básico (configure OPENAI_API_KEY para IA)."
            
            return jsonify({'message': msg, 'success': True})
        else:
            return jsonify({'message': f'Erro: {agent.error_message}', 'success': False}), 500
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Erro: {str(e)}', 'success': False}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'Faça uma pergunta', 'image': None})
        
        result = agent.run_query(query)
        return jsonify({'response': result.get('response', 'Sem resposta'), 'image': result.get('image', None)})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f'Erro: {str(e)}', 'image': None})

@app.route('/logs')
def get_logs():
    return jsonify({'logs': log_stream.logs})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'openai': 'configured' if API_KEY else 'missing',
        'dataset': 'loaded' if agent.df is not None else 'empty'
    })

@app.errorhandler(413)
def too_large(error):
    return jsonify({'message': 'Arquivo muito grande (max 50MB)', 'success': False}), 413

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor na porta {port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
