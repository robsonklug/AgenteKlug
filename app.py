from flask import Flask, request, render_template_string, session
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
import uuid
from agent import DataAnalysisAgent

from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Adiciona CORS ao aplicativo Flask app
app.secret_key = os.urandom(24) # Chave secreta para gerenciar sessões
app.config["UPLOAD_FOLDER"] = "uploads"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Template HTML para a interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise de Dados com IA</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #0056b3; text-align: center; margin-bottom: 30px; }
        .upload-section, .chat-section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
        .upload-section h2, .chat-section h2 { color: #0056b3; margin-top: 0; }
        input[type="file"], input[type="text"], button { padding: 10px 15px; border-radius: 5px; border: 1px solid #ccc; font-size: 16px; }
        input[type="file"] { background-color: #e9ecef; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; transition: background-color 0.3s ease; }
        button:hover { background-color: #0056b3; }
        .chat-history { border: 1px solid #eee; padding: 15px; max-height: 400px; overflow-y: auto; background-color: #fff; border-radius: 5px; margin-top: 20px; }
        .message { margin-bottom: 10px; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #e0f7fa; text-align: right; }
        .agent-message { background-color: #e8f5e9; text-align: left; }
        .agent-message img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
        .error-message { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Análise de Dados com IA</h1>
# Importar seu agente
from data_analysis_agent import DataAnalysisAgent

        <div class="upload-section">
            <h2>1. Carregar Arquivo CSV</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="csv_file" accept=".csv" required>
                <button type="submit">Carregar CSV</button>
            </form>
            {% if message %}<p>{{ message }}</p>{% endif %}
        </div>
# Configurar matplotlib para ambiente sem GUI
plt.switch_backend('Agg')
load_dotenv()

        {% if session.get('csv_loaded') %}
        <div class="chat-section">
            <h2>2. Faça sua Pergunta</h2>
            <form action="/ask" method="post">
                <input type="text" name="question" placeholder="Ex: Qual a média da coluna Amount?" size="50" required>
                <button type="submit">Perguntar</button>
            </form>
            <div class="chat-history">
                {% for msg in session.get('chat_history', []) %}
                    <div class="message {{ 'user-message' if msg.sender == 'user' else 'agent-message' }}">
                        <strong>{{ msg.sender }}:</strong> {{ msg.text | safe }}
                    </div>
                {% endfor %}
            </div>
        </div>
        {% else %}
        <div class="chat-section">
            <p>Por favor, carregue um arquivo CSV para começar a fazer perguntas.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Dicionário para armazenar instâncias do agente por sessão
# Em um ambiente de produção, isso seria gerenciado de forma mais robusta (ex: banco de dados, cache distribuído)
user_agents = {}
# Instância global do agente
agent = DataAnalysisAgent()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, message=session.get('message'))
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        session['message'] = 'Nenhum arquivo enviado.'
        return render_template_string(HTML_TEMPLATE, message=session['message'])

    file = request.files['csv_file']
    if file.filename == '':
        session['message'] = 'Nenhum arquivo selecionado.'
        return render_template_string(HTML_TEMPLATE, message=session['message'])

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4()) # Gerar um ID de sessão único para o usuário
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id + '_' + filename)
        file.save(upload_path)

        agent = DataAnalysisAgent()
        try:
            agent.load_csv(upload_path)
            user_agents[session_id] = agent
            session['session_id'] = session_id
            session['csv_loaded'] = True
            session['chat_history'] = []
            session['message'] = f'Arquivo {filename} carregado com sucesso! Agora você pode fazer perguntas.'
        except Exception as e:
            session['message'] = f'Erro ao processar o CSV: {e}'
            session['csv_loaded'] = False
            if session_id in user_agents: del user_agents[session_id]
            if os.path.exists(upload_path): os.remove(upload_path)

        return render_template_string(HTML_TEMPLATE, message=session['message'], csv_loaded=session.get('csv_loaded'))
    else:
        session['message'] = 'Tipo de arquivo não permitido. Por favor, envie um arquivo CSV.'
        return render_template_string(HTML_TEMPLATE, message=session['message'])
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'Nenhum arquivo enviado'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'Arquivo inválido'}), 400

        if file and file.filename.endswith('.csv'):
            # Salvar temporariamente (usar /tmp no Render)
            file_path = f"/tmp/{file.filename}"
            file.save(file_path)
            
            # Carregar no agente
            if agent.load_csv(file_path):
                # Remover arquivo temporário após carregar
                try:
                    os.remove(file_path)
                except:
                    pass  # Ignorar se não conseguir remover
                
                return jsonify({
                    'message': f'CSV carregado com sucesso! {len(agent.df)} linhas e {len(agent.df.columns)} colunas.',
                    'success': True
                })
            else:
                return jsonify({'message': 'Erro ao processar o arquivo CSV'}), 500
        
        return jsonify({'message': 'Apenas arquivos .csv são aceitos'}), 400
    
    except Exception as e:
        print(f"Erro no upload: {e}")
        return jsonify({'message': f'Erro interno: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form['question']
    session_id = session.get('session_id')

    if not session_id or session_id not in user_agents:
        session['message'] = 'Por favor, carregue um arquivo CSV primeiro.'
        return render_template_string(HTML_TEMPLATE, message=session['message'], csv_loaded=session.get('csv_loaded'))

    agent = user_agents[session_id]
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'Por favor, faça uma pergunta.'})
        
        # Usar o agente para processar a consulta
        response = agent.run_query(query)
        
        return jsonify({'response': response})

    # Adicionar pergunta do usuário ao histórico
    chat_history = session.get('chat_history', [])
    chat_history.append({'sender': 'user', 'text': user_question})
    session['chat_history'] = chat_history
    except Exception as e:
        print(f"Erro na consulta: {e}")
        return jsonify({'response': f'Erro ao processar a consulta: {str(e)}'})

@app.route('/info')
def get_info():
    """Endpoint para obter informações sobre o dataset carregado"""
    try:
        agent_response = agent.run_query(user_question)
        # Adicionar resposta do agente ao histórico
        chat_history.append({'sender': 'agent', 'text': agent_response})
        session['chat_history'] = chat_history
        session['message'] = None # Limpar mensagem de upload
        info = agent.get_dataframe_info()
        return jsonify(info)
    except Exception as e:
        error_msg = f"Ocorreu um erro ao processar a sua pergunta: {e}"
        chat_history.append({'sender': 'agent', 'text': f'<span class="error-message">{error_msg}</span>'})
        session['chat_history'] = chat_history
        session['message'] = None

    return render_template_string(HTML_TEMPLATE, message=session['message'], csv_loaded=session.get('csv_loaded'))
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Aplicação funcionando normalmente',
        'agent_status': 'loaded' if agent.df is not None else 'no_data'
    })

# Tratamento de erros
@app.errorhandler(413)
def too_large(e):
    return jsonify({'message': 'Arquivo muito grande. Máximo 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'message': 'Erro interno do servidor'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'message': 'Endpoint não encontrado'}), 404

# Configuração para produção
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Render usa a variável PORT automaticamente
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

# Configuração para produção
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
