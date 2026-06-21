# Importa as classes necessárias da biblioteca transformers para tokenização e treinamento do modelo
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Carrega o modelo GPT-2 pré-treinado para geração de texto
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Carrega o tokenizador correspondente ao modelo GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Preparar conjunto de dados
texts = ["Contrato de compra e venda...", "Ação de execução de título..."]
# Prepara o conjunto de dados para treinamento
# - texts: Lista de textos que será usada como dados de entrada para o modelo
# - return_tensors: Define o formato de saída como tensores PyTorch
# - max_length: Limita o tamanho máximo de cada sequência
# - truncation: Trunca textos que excedem o tamanho máximo
# - padding: Adiciona preenchimento para igualar o comprimento das sequências
inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True)

# # Configura os argumentos para o treinamento do modelo
# - output_dir: Diretório onde os resultados do treinamento serão salvos
# - num_train_epochs: Número de épocas para o treinamento
# - per_device_train_batch_size: Tamanho do lote usado por dispositivo (ex.: GPU)
# - save_steps: Salva o modelo a cada N passos
# - save_total_limit: Número máximo de checkpoints salvos
# - logging_dir: Diretório para salvar os logs do treinamento
training_args = TrainingArguments(
	output_dir="./results",
	num_train_epochs=3,
	per_device_train_batch_size=4,
	save_steps=10_000,
	save_total_limit=2,
	logging_dir="./logs",
)

# Cria o objeto Trainer para realizar o treinamento
# - model: O modelo que será treinado
# - args: Configurações do treinamento
# - train_dataset: O conjunto de dados preparado para o treinamento
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=inputs,
)

# Inicia o processo de treinamento
trainer.train()