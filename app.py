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
import warnings

# Configurações
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

class DataAnalysisAgent:
    def __init__(self):
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.agent_executor = None

    def load_csv(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            print(f"CSV carregado: {len(self.df)} linhas e {len(self.df.columns)} colunas.")
            self._initialize_agent()
            return True
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return False

    def _initialize_agent(self):
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        try:
            self.agent_executor = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="openai-tools",
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60
            )
            print("Agente inicializado com sucesso!")
        except Exception as e:
            print(f"Erro na inicialização: {e}")
            try:
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    verbose=True,
                    allow_dangerous_code=True
                )
                print("Agente inicializado em modo básico")
            except Exception as e2:
                print(f"Falha total: {e2}")
                raise

    def run_query(self, query: str) -> str:
        if self.agent_executor is None:
            return "Agente não inicializado. Carregue um CSV primeiro."
        
        try:
            enhanced_query = f"""
Dataset: {len(self.df)} linhas, {len(self.df.columns)} colunas
Colunas: {', '.join(self.df.columns[:10])}{'...' if len(self.df.columns) > 10 else ''}

Pergunta: {query}
"""
            
            result = self.agent_executor.invoke({"input": enhanced_query})
            return result.get("output", "Sem resposta disponível.")
        except Exception as e:
            print(f"Erro na consulta: {e}")
            return self._handle_simple_queries(query)

    def _handle_simple_queries(self, query: str) -> str:
        query_lower = query.lower()
        
        try:
            if any(word in query_lower for word in ['primeiras', 'first', 'head']):
                n = self._extract_number(query, default=5)
                return f"Primeiras {n} linhas:\n{self.df.head(n).to_string()}"
            elif any(word in query_lower for word in ['colunas', 'columns']):
                return f"Colunas: {list(self.df.columns)}"
            elif any(word in query_lower for word in ['shape', 'tamanho']):
                return f"Dimensões: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas"
            elif any(word in query_lower for word in ['info', 'informações']):
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                return buffer.getvalue()
            elif 'describe' in query_lower:
                return f"Estatísticas:\n{self.df.describe().to_string()}"
            else:
                return ("Consultas disponíveis:\n"
                       "- primeiras linhas\n"
                       "- colunas\n"
                       "- informações\n"
                       "- tamanho\n"
                       "- describe")
        except Exception as e:
            return f"Erro: {e}"

    def _extract_number(self, text: str, default: int = 5) -> int:
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return min(int(numbers[0]), 20)
        return default

# Instância global
agent = DataAnalysisAgent()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analisador de Dados IA</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .section { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .upload-area { border: 2px dashed #007bff; padding: 20px; text-align: center; border-radius: 8px; }
            .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; background: white; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #007bff; color: white; margin-left: 20%; }
            .agent { background: #28a745; color: white; margin-right: 20%; }
            .system { background: #6c757d; color: white; text-align: center; }
            input, button { padding: 10px; margin: 5px; }
            button { background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            pre { white-space: pre-wrap; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🤖 Analisador de Dados com IA</h1>
        
        <div class="section">
            <h3>Upload CSV</h3>
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".csv">
                <button onclick="uploadFile()">Carregar</button>
            </div>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="section">
            <h3>Chat</h3>
            <div class="chat-box" id="chatBox">
                <div class="message system">Carregue um CSV para começar</div>
            </div>
            <div>
                <input type="text" id="queryInput" placeholder="Faça uma pergunta..." style="width: 70%;">
                <button onclick="sendQuery()">Enviar</button>
            </div>
        </div>

        <script>
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Selecione um arquivo!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                document.getElementById('uploadStatus').innerHTML = 
                    `<div style="color: ${result.success ? 'green' : 'red'}">${result.message}</div>`;
                
                if (result.success) {
                    addMessage('Sistema', result.message);
                }
            } catch (error) {
                document.getElementById('uploadStatus').innerHTML = 
                    `<div style="color: red">Erro: ${error.message}</div>`;
            }
        }

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            
            if (!query) return;
            
            addMessage('Você', query);
            input.value = '';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const result = await response.json();
                addMessage('IA', result.response);
            } catch (error) {
                addMessage('Erro', 'Falha na comunicação: ' + error.message);
            }
        }

        function addMessage(sender, message) {
            const chatBox = document.getElementById('chatBox');
            const div = document.createElement('div');
            
            let className = 'message ';
            if (sender === 'Você') className += 'user';
            else if (sender === 'IA') className += 'agent';
            else className += 'system';
            
            div.className = className;
            div.innerHTML = `<strong>${sender}:</strong><pre>${message}</pre>`;
            
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendQuery();
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
            file_path = f"/tmp/{file.filename}"
            file.save(file_path)
            
            if agent.load_csv(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
                
                return jsonify({
                    'message': f'CSV carregado! {len(agent.df)} linhas, {len(agent.df.columns)} colunas',
                    'success': True
                })
            else:
                return jsonify({'message': 'Erro ao processar CSV', 'success': False}), 500
        
        return jsonify({'message': 'Apenas arquivos .csv', 'success': False}), 400
    except Exception as e:
        print(f"Erro no upload: {e}")
        return jsonify({'message': f'Erro: {str(e)}', 'success': False}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'Faça uma pergunta.'})
        
        response = agent.run_query(query)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Erro na consulta: {e}")
        return jsonify({'response': f'Erro: {str(e)}'})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'agent': 'loaded' if agent.df is not None else 'empty'
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
