from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Importar seu agente
from data_analysis_agent import DataAnalysisAgent

# Configurar matplotlib para ambiente sem GUI
plt.switch_backend('Agg')
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Instância global do agente
agent = DataAnalysisAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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

@app.route('/info')
def get_info():
    """Endpoint para obter informações sobre o dataset carregado"""
    try:
        info = agent.get_dataframe_info()
        return jsonify(info)
    except Exception as e:
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

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Configuração para produção
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
