# Importa a biblioteca do cliente para o Google Cloud Natural Language API
from google.cloud import language_v1

# Inicializa o cliente para interagir com a API de análise de linguagem
client = language_v1.LanguageServiceClient()

# Define o texto que será analisado para sentimentos
text_content = "A inovação tecnológica é essencial para o progresso da sociedade."

# Cria um objeto de documento que contém o texto e especifica o tipo de conteúdo como texto simples (plain text)
document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)

# Realiza a análise de sentimento no texto fornecido e armazena os resultados 
# - request: Dicionário que especifica o documento a ser analisado 
# - document_sentiment: Retorna os detalhes do sentimento, como score e magnitude
sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

# Exibe o score do sentimento, indicando se o texto é positivo (valores próximos a 1), negativo (valores próximos a -1) ou neutro (próximo a 0)
print("Score de Sentimento:", sentiment.score)

# Exibe a magnitude do sentimento, que indica a intensidade emocional do texto, independentemente de ser positiva ou negativa
print("Magnitude do Sentimento:", sentiment.magnitude)