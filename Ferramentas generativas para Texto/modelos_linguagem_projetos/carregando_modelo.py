# Importa as classes para trabalhar com modelos e tokenizadores
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define o nome de um modelo treinado para português
model_name = "pierreguillou/gpt2-small-portuguese"  # Modelo GPT-2 ajustado para português

# Carrega o modelo e o tokenizador correspondentes
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define o prompt inicial que será usado como entrada para o modelo
prompt = """
Escreva um parágrafo em português sobre como a inteligência artificial está transformando o setor de saúde.
Foque em exemplos como diagnósticos precisos, análise de dados médicos e personalização de tratamentos.
"""

# Tokeniza o prompt, convertendo-o em tensores que o modelo pode processar
# - return_tensors="pt": Retorna os tokens no formato de tensor PyTorch
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Gera texto com base no prompt tokenizado
outputs = model.generate(
    inputs,
    max_length=200,  # Limita o comprimento da saída
    num_return_sequences=1,  # Gera uma sequência
    temperature=0.9,  # Controla a criatividade do texto
    do_sample=True,  # Ativa amostragem aleatória
    pad_token_id=tokenizer.eos_token_id,  # Define o token de preenchimento
    top_p=0.95,
    eos_token_id=None
)

# Decodifica os tokens gerados de volta para texto legível
# - skip_special_tokens=True: Remove tokens especiais, como "<|endoftext|>"
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exibe o texto gerado
print(generated_text)

# Prompt para gerar um resumo
prompt = "Resuma o seguinte texto: A reciclagem é importante porque reduz resíduos, economiza energia e protege o meio ambiente."

inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs, 
    max_length=300, 
    num_return_sequences=1, 
    temperature=0.2,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

