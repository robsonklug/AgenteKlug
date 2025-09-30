import os
import pandas as pd
from flask import Flask, request, render_template_string
from langchain_openai import ChatOpenAI

# =========================
# Configuração da API
# =========================
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
IA_ENABLED = bool(OPENAI_KEY)

if IA_ENABLED:
    print(f"✅ OPENAI_API_KEY detectada. Tamanho: {len(OPENAI_KEY)}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)
else:
    print("⚠️ Nenhuma chave encontrada. Usando modo básico (pandas).")
    llm = None  # IA não disponível

# =========================
# Flask App
# =========================
app = Flask(__name__)

df_global = None  # DataFrame global

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Agente CSV</title>
</head>
<body>
  <h1>Agente CSV</h1>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Carregar CSV">
  </form>
  <p>{{ status }}</p>
  <form method="POST" action="/query">
    <input type="text" name="pergunta" placeholder="Faça sua pergunta">
    <input type="submit" value="Perguntar">
  </form>
  <p>{{ resposta }}</p>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global df_global
    status = ""
    resposta = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            df_global = pd.read_csv(file)
            if IA_ENABLED:
                status = f"CSV carregado: {df_global.shape[0]} linhas x {df_global.shape[1]} colunas. Modo IA habilitado ✅"
            else:
                status = f"CSV carregado: {df_global.shape[0]} linhas x {df_global.shape[1]} colunas. Modo básico (sem IA)."

    return render_template_string(HTML_TEMPLATE, status=status, resposta=resposta)

@app.route("/query", methods=["POST"])
def query():
    global df_global
    pergunta = request.form["pergunta"]
    resposta = ""

    if df_global is None:
        resposta = "Nenhum CSV carregado."
    else:
        if IA_ENABLED and llm:
            resposta = llm.invoke(pergunta).content
        else:
            # Fallback simples com pandas
            resposta = f"(Modo básico) Você perguntou: {pergunta}. Não há IA disponível."

    return render_template_string(HTML_TEMPLATE, status="", resposta=resposta)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
