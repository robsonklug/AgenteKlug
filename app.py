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
import sys

# Configurações
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Log de inicialização
print("=" * 60)
print("INICIANDO APLICAÇÃO")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"OPENAI_API_KEY configurada: {'SIM ✓' if os.environ.get('OPENAI_API_KEY') else 'NÃO ✗'}")
if os.environ.get('OPENAI_API_KEY'):
    key = os.environ.get('OPENAI_API_KEY')
    print(f"Primeiros caracteres da chave: {key[:15]}...")
print("=" * 60)

class DataAnalysisAgent:
    def __init__(self):
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  AVISO: OPENAI_API_KEY não configurada!")
            print("   Configure a chave no Render Dashboard > Service > Environment")
        
        self.llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini",
            openai_api_key=api_key
        )
        self.agent_executor = None

    def load_csv(self, file_path):
        print(f"\n{'='*60}")
        print(f"CARREGANDO CSV: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Passo 1: Carregar CSV
            print("Passo 1: Lendo arquivo CSV...")
            self.df = pd.read_csv(file_path)
            print(f"✓ CSV lido com sucesso!")
            print(f"  - Linhas: {len(self.df)}")
            print(f"  - Colunas: {len(self.df.columns)}")
            print(f"  - Colunas: {list(self.df.columns[:5])}{'...' if len(self.df.columns) > 5 else ''}")
            
            # Passo 2: Verificar chave OpenAI
            print("\nPasso 2: Verificando configuração OpenAI...")
            if not os.environ.get("OPENAI_API_KEY"):
                print("✗ ERRO: OPENAI_API_KEY não encontrada!")
                print("  A aplicação funcionará apenas com consultas básicas.")
                return True  # Retorna True para permitir uso básico
            else:
                print("✓ OPENAI_API_KEY encontrada")
            
            # Passo 3: Inicializar agente
            print("\nPasso 3: Inicializando agente LangChain...")
            self._initialize_agent()
            
            print(f"\n{'='*60}")
            print("CSV CARREGADO COM SUCESSO!")
            print(f"{'='*60}\n")
            return True
            
        except Exception as e:
            print(f"\n✗ ERRO DETALHADO:")
            print(f"  Tipo: {type(e).__name__}")
            print(f"  Mensagem: {str(e)}")
            import traceback
            print(f"  Traceback:\n{traceback.format_exc()}")
            return False

    def _initialize_agent(self):
        if self.df is None:
            raise ValueError("DataFrame não carregado.")

        try:
            print("  → Tentando inicialização completa...")
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
            print("  ✓ Agente LangChain inicializado com SUCESSO!")
            
        except Exception as e:
            print(f"  ✗ Falha na inicialização completa: {str(e)}")
            
            try:
                print("  → Tentando inicialização básica...")
                self.agent_executor = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.df,
                    verbose=True,
                    allow_dangerous_code=True
                )
                print("  ✓ Agente inicializado em MODO BÁSICO")
                
            except Exception as e2:
                print(f"  ✗ Falha na inicialização básica: {str(e2)}")
                print("  ⚠️  Continuando SEM agente LangChain")
                print("     Apenas consultas básicas estarão disponíveis")
                self.agent_executor = None

    def run_query(self, query: str) -> str:
        print(f"\n{'='*60}")
        print(f"PROCESSANDO CONSULTA: {query}")
        print(f"{'='*60}")
        
        if self.agent_executor is None:
            print("⚠️  Usando modo de consultas básicas (agente não disponível)")
            return self._handle_simple_queries(query)
        
        try:
            print("→ Usando agente LangChain...")
            enhanced_query = f"""
Dataset: {len(self.df)} linhas, {len(self.df.columns)} colunas
Colunas: {', '.join(self.df.columns[:10])}{'...' if len(self.df.columns) > 10 else ''}

Pergunta: {query}
"""
            
            result = self.agent_executor.invoke({"input": enhanced_query})
            response = result.get("output", "Sem resposta disponível.")
            print(f"✓ Resposta gerada com sucesso")
            return response
            
        except Exception as e:
            print(f"✗ Erro no agente LangChain: {str(e)}")
            print("→ Tentando modo básico...")
            return self._handle_simple_queries(query)

    def _handle_simple_queries(self, query: str) -> str:
        query_lower = query.lower()
        print(f"  Processando consulta básica...")
        
        try:
            if any(word in query_lower for word in ['primeiras', 'first', 'head']):
                n = self._extract_number(query, default=5)
                return f"Primeiras {n} linhas:\n{self.df.head(n).to_string()}"
            
            elif any(word in query_lower for word in ['colunas', 'columns']):
                return f"Colunas do dataset:\n{list(self.df.columns)}"
            
            elif any(word in query_lower for word in ['shape', 'tamanho']):
                return f"Dimensões: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas"
            
            elif any(word in query_lower for word in ['info', 'informações']):
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                return buffer.getvalue()
            
            elif 'describe' in query_lower or 'estatísticas' in query_lower:
                return f"Estatísticas descritivas:\n{self.df.describe().to_string()}"
            
            elif any(word in query_lower for word in ['nulos', 'null', 'missing']):
                nulls = self.df.isnull().sum()
                return f"Valores nulos:\n{nulls[nulls > 0].to_string()}"
            
            else:
                return ("Consultas básicas disponíveis:\n"
                       "- 'primeiras N linhas'\n"
                       "- 'colunas'\n"
                       "- 'informações'\n"
                       "- 'tamanho'\n"
                       "- 'describe/estatísticas'\n"
                       "- 'valores nulos'\n\n"
                       "Configure OPENAI_API_KEY para consultas avançadas com IA.")
        except Exception as e:
            print(f"  ✗ Erro na consulta básica: {str(e)}")
            return f"Erro ao processar consulta: {e}"

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
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>🤖 Analisador de Dados com IA</h1>
        <p>Powered by GPT-4o-mini + LangChain</p>
        
        <div class="section">
            <h3>📁 Upload CSV</h3>
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".csv">
                <button onclick="uploadFile()">Carregar CSV</button>
            </div>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="section">
            <h3>💬 Chat com IA</h3>
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
                alert('Selecione um arquivo CSV!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('uploadStatus').innerHTML = 
                '<div class="status">Carregando...</div>';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                document.getElementById('uploadStatus').innerHTML = 
                    `<div class="status ${result.success ? 'success' : 'error'}">${result.message}</div>`;
                
                if (result.success) {
                    addMessage('Sistema', result.message);
                }
            } catch (error) {
                document.getElementById('uploadStatus').innerHTML = 
                    `<div class="status error">Erro: ${error.message}</div>`;
            }
        }

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            
            if (!query) return;
            
            addMessage('Você', query);
            input.value = '';
            addMessage('Sistema', '⏳ Processando...');
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const result = await response.json();
                
                // Remove a mensagem de "Processando..."
                const chatBox = document.getElementById('chatBox');
                chatBox.removeChild(chatBox.lastChild);
                
                addMessage('IA', result.response);
            } catch (error) {
                addMessage('Erro', 'Falha: ' + error.message);
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
    print(f"\n{'='*60}")
    print("ENDPOINT /upload CHAMADO")
    print(f"{'='*60}")
    
    try:
        if 'file' not in request.files:
            print("✗ Nenhum arquivo na requisição")
            return jsonify({'message': 'Nenhum arquivo enviado', 'success': False}), 400

        file = request.files['file']
        if file.filename == '':
            print("✗ Nome de arquivo vazio")
            return jsonify({'message': 'Arquivo inválido', 'success': False}), 400

        if file and file.filename.endswith('.csv'):
            print(f"✓ Arquivo recebido: {file.filename}")
            file_path = f"/tmp/{file.filename}"
            file.save(file_path)
            print(f"✓ Arquivo salvo em: {file_path}")
            
            if agent.load_csv(file_path):
                try:
                    os.remove(file_path)
                    print(f"✓ Arquivo temporário removido")
                except:
                    print(f"⚠️  Não foi possível remover arquivo temporário")
                
                # Determinar status
                if agent.agent_executor:
                    msg = f"✓ CSV carregado com IA! {len(agent.df)} linhas, {len(agent.df.columns)} colunas"
                else:
                    msg = f"⚠️  CSV carregado em MODO BÁSICO. {len(agent.df)} linhas, {len(agent.df.columns)} colunas. Configure OPENAI_API_KEY para usar IA."
                
                print(msg)
                return jsonify({'message': msg, 'success': True})
            else:
                print("✗ Falha ao carregar CSV no agente")
                # Fallback: carregar apenas o dataframe
                try:
                    temp_df = pd.read_csv(file_path)
                    agent.df = temp_df
                    agent.agent_executor = None
                    
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    
                    msg = f"⚠️  CSV carregado PARCIALMENTE. {len(agent.df)} linhas, {len(agent.df.columns)} colunas. Apenas consultas básicas."
                    print(msg)
                    return jsonify({'message': msg, 'success': True})
                except Exception as e:
                    print(f"✗ Erro no fallback: {str(e)}")
                    return jsonify({'message': f'Erro ao processar CSV: {str(e)}', 'success': False}), 500
        
        print("✗ Arquivo não é CSV")
        return jsonify({'message': 'Apenas arquivos .csv são aceitos', 'success': False}), 400
        
    except Exception as e:
        print(f"✗ ERRO CRÍTICO no upload:")
        print(f"  Tipo: {type(e).__name__}")
        print(f"  Mensagem: {str(e)}")
        import traceback
        print(f"  Traceback:\n{traceback.format_exc()}")
        return jsonify({'message': f'Erro: {str(e)}', 'success': False}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'Por favor, faça uma pergunta.'})
        
        response = agent.run_query(query)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"✗ Erro em /ask: {str(e)}")
        return jsonify({'response': f'Erro: {str(e)}'})

@app.route('/health')
def health():
    status = {
        'status': 'ok',
        'openai_key': 'configured' if os.environ.get('OPENAI_API_KEY') else 'missing',
        'agent': 'loaded' if agent.df is not None else 'empty',
        'langchain_agent': 'active' if agent.agent_executor else 'inactive'
    }
    print(f"\nHealth check: {status}")
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"INICIANDO SERVIDOR NA PORTA {port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
