# Importa o cliente para o IBM Watson Natural Language Understanding 
from ibm_watson import NaturalLanguageUnderstandingV1

# Importa as classes necessárias para configurar as features de análise, incluindo análise de emoções
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

# Inicializa o cliente para o IBM Watson Natural Language Understanding, configurando a versão da API, 
# a chave de API para autenticação e a URL do serviço
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    iam_apikey='sua-chave-api',
    url='sua-url'
)

# Realiza a análise de emoções no texto fornecido 
# - text: Define o texto que será analisado 
# - features: Especifica o tipo de análise a ser realizada (emoções no documento neste caso)
response = natural_language_understanding.analyze(
    text="A tecnologia blockchain está revolucionando o setor financeiro.",
    features=Features(emotion=EmotionOptions(document=True))
).get_result()

# Exibe as emoções detectadas no texto, retornadas como um dicionário com os níveis de emoção (alegria, tristeza, etc.)
print(response['emotion']['document']['emotion'])