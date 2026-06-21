# Importa o pipeline da biblioteca transformers, que permite usar modelos pré-treinados para várias tarefas de NLP (Processamento de Linguagem Natural)
from transformers import pipeline

# Cria um pipeline de geração de texto utilizando o modelo pré-treinado GPT-2
gerador_texto = pipeline('text-generation', model='gpt2')

# Define o prompt inicial que será usado como entrada para o modelo
prompt = "A inteligência artificial está transformando"

# Gera texto com base no prompt fornecido, limitando o resultado a 50 tokens e gerando apenas 1 sequência
resultado = gerador_texto(prompt, max_length=50, num_return_sequences=1)

print("RESULTADO: ", resultado[0])

# Imprime o texto gerado pelo modelo, acessando o campo 'generated_text' da primeira sequência retornada
print(resultado[0]['generated_text'])
