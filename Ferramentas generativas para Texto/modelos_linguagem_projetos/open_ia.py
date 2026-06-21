# Importa a biblioteca OpenAI, que fornece acesso à API de modelos de linguagem, como o GPT
import OpenAI

# Define a chave de API necessária para autenticação na plataforma OpenAI 
OpenAI.api_key = 'sua-chave-api'

# Define o prompt que será enviado para o modelo, contendo a instrução ou pergunta
prompt = "Explique a importância da reciclagem para o meio ambiente."

# Envia uma solicitação para a API da OpenAI para gerar uma resposta baseada no prompt 
# - engine: Define o modelo que será utilizado (text-davinci-003, neste caso) 
# - prompt: O texto de entrada que orienta o modelo 
# - max_tokens: Limita o número máximo de tokens na resposta gerada (100 tokens neste caso) 
# - temperature: Controla o nível de criatividade da resposta (valores mais altos geram respostas mais criativas)
resposta = OpenAI.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7)

# Exibe a resposta gerada pelo modelo, acessando o texto da primeira escolha retornada e removendo espaços extras
print(resposta.choices[0].text.strip())