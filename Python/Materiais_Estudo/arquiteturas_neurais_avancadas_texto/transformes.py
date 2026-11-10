# pip install torch==2.8.0 transformers==4.56.1
from transformers import pipeline # Ferramenta que facilita o uso de modelos prontos de IA, como geração de texto
from transformers.utils import logging
import torch  # Biblioteca que ajuda a usar a GPU ou CPU para acelerar os cálculos de IA
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.set_verbosity_error()

# Define automaticamente o dispositivo para execução: usa GPU (CUDA) se disponível, senão MPS (Apple Silicon), ou CPU como fallback
if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = -1

# Define o texto inicial que será usado como prompt para o modelo continuar
prompt = "i love you"

# Cria um gerador de texto usando o modelo gpt2 da comunidade OpenAI no Hugging Face, já escolhendo o melhor dispositivo (GPU ou CPU)
generator = pipeline("text-generation", model="openai-community/gpt2", device=device)

# Gera a continuação do texto a partir do prompt, com até 50 tokens adicionais
resposta = generator(prompt, max_length=20)

# Exibe o texto gerado na tela
print('Resposta Inglês: ', resposta[0]['generated_text'])

"""# Sobre o Treinamento dos Modelos GPT
Grandes modelos como o **GPT-2**, desenvolvidos pela OpenAI, foram treinados usando uma enorme coleção de textos em inglês disponíveis na internet — 
cerca de 8 milhões de páginas. O objetivo central desse treinamento era simples e poderoso: ensinar o modelo a **prever a próxima palavra** em uma sequência de texto.
O GPT-2 faz isso analisando grandes volumes de frases e aprendendo padrões de como as palavras costumam aparecer juntas.
Esse método de treinamento — prever sempre o próximo termo — é a base que permitiu a evolução de modelos cada vez maiores e mais capazes, como os GPTs mais recentes, 
que seguem a mesma lógica fundamental de linguagem e que popularizaram os LLM's
Vimos então que o GPT-2 foi treinado em dados em inglês, e ao testar acima foi possível ver que é funcional para a linguagem, completando o texto fornecido. 
A pergunta que fica é o que aconteceria se testarmos o texto em português?
"""
prompt = "Eu amo você"

generator = pipeline("text-generation", model="openai-community/gpt2", device=device)

resposta = generator(prompt, max_length=20)

print('Resposta português: ', resposta[0]['generated_text'])

"""Vimos então que o resultado não é tão satisfatório assim, demonstrando a grande necessidade de 
um amplo domínio de dados para garantir modelos de alta performance em várias línguas, como os modelos atuais.
# Sobre o Tamanho dos Modelos de Linguagem
O tamanho de um modelo de linguagem refere-se à quantidade de parâmetros que ele possui e ao espaço que ocupa no disco após ser baixado. 
Em geral, **modelos maiores são mais poderosos e produzem textos mais sofisticados**, mas também consomem mais memória e processamento. Já modelos menores são mais rápidos e leves, ideais para testes e uso em máquinas comuns.
O modelo **GPT-2** da OpenAI, por exemplo, possui cerca de 124 milhões de parâmetros. Esse tamanho é suficiente para gerar textos de boa qualidade, ao mesmo tempo em que ainda é viável para ser rodado em computadores pessoais.
Abaixo, mostramos como verificar o tamanho ocupado pelo modelo GPT-2 após o download:
"""

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
num_params = sum(p.numel() for p in model.parameters())
print(f"Número de parâmetros do modelo GPT-2: {num_params:,}")

"""# Modelos Recentes e sua Escala
Nos últimos anos, os LLMs evoluíram para tamanhos impressionantes. Um dos mais notáveis é o **DeepSeek-V3**, que conta com **671 bilhões de parâmetros**.
Antes dele, o **DeepSeek LLM** também já oferecia **67 bilhões de parâmetros**, tendo sido treinado do zero com cerca de **2 trilhões de tokens** de texto em inglês e chinês.
Para comparação, o **GPT‑2** da OpenAI, lançado em 2019, possui 124 milhões de parâmetros e foi treinado com cerca de **10 bilhões de tokens** de texto em inglês.
Esses números destacam como os modelos atuais cresceram em capacidade e dados, comparado aos 124 M de parâmetros do GPT‑2. Isso possibilita habilidades de linguagem, 
raciocínio e multimodalidade muito mais avançadas — embora exijam hardware e energia muito acima dos modelos menores.
"""
# Dados do GPT-2
gpt2_params = 124_000_000         # 124 milhões de parâmetros
gpt2_tokens = 10_000_000_000      # 10 bilhões de tokens

# Dados do DeepSeek-V3
deepseek_v3_params = 671_000_000_000   # 671 bilhões de parâmetros
deepseek_v3_tokens = 8_100_000_000_000  # 8,1 trilhões de tokens

params_factor = deepseek_v3_params / gpt2_params
tokens_factor = deepseek_v3_tokens / gpt2_tokens

print(f"DeepSeek‑V3 tem {int(params_factor)}x mais parâmetros que o GPT‑2.")
print(f"DeepSeek‑V3 foi treinado com {int(tokens_factor)}x mais tokens que o GPT‑2.")

"""## O que são parâmetros?
Parâmetros são os números ajustáveis dentro de um modelo de IA — como se fossem “botões” internos que guardam o que o modelo aprendeu.  
Quanto mais parâmetros, a tendência é que maior seja a capacidade do modelo para reconhecer padrões e gerar textos sofisticados. Contudo, também aumentam o custo de armazenamento, 
processamento e energia necessários para treinar e usar o modelo, isso de maneira drástica.
"""

def memoria_parametros(num_params, nr_bits=32):
    """
    Calcula a memória ocupada pelos parâmetros de um modelo PyTorch,
    usando a fórmula memory = (nr_bits / 8) * nr_params (em bytes).
    """
    bytes_por_param = nr_bits / 8
    total_bytes = num_params * bytes_por_param
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    print(f"Número de parâmetros: {num_params:,}")
    print(f"Ocupação na memória ({nr_bits}-bit): {total_mb:.2f} MB ({total_gb:.3f} GB)")


memoria_parametros(gpt2_params)

""" **Atenção:**  
 O cálculo acima mostra apenas a memória necessária para armazenar os parâmetros do modelo enquanto ele está carregado na memória.  
Durante o **treinamento**, o uso real de memória é muito maior, pois também precisamos guardar os gradientes (para o cálculo e atualização dos pesos) e os estados internos do otimizador (como o Adam), que multiplicam o consumo de VRAM/RAM.  
Por isso, treinar grandes modelos exige muito mais recursos do que apenas armazenar seus parâmetros.
Vamos ver uma exemplificação disso abaixo
"""

def memoria_total_llm(n_params,
                      peso_bytes=4,
                      grad_bytes=4,
                      adam_bytes=4,
                      num_adam_states=2
                      ):
    memoria_pesos = n_params * peso_bytes
    memoria_grad = n_params * grad_bytes
    memoria_adam = n_params * num_adam_states * adam_bytes
    total = memoria_pesos + memoria_grad + memoria_adam
    divisor = 1024 ** 3

    print(f"Memória dos pesos: {memoria_pesos/divisor:.3f} GB")
    print(f"Memória dos gradientes: {memoria_grad/divisor:.3f} GB")
    print(f"Memória do otimizador (Adam): {memoria_adam/divisor:.3f} GB")
    print(f"Memória total estimada: {total/divisor:.3f} GB")

memoria_total_llm(gpt2_params)

"""Nesse primeiro momento parece "pouco" Mas isso é por que treinamos um modelo pequeno, vamos fazer a mesma simulação para um modelo 7B"""
memoria_total_llm(7_000_000_000) # 7 bilhões de parâmetros

"""## Considerações sobre Memória e o Treinamento de LLMs
Com as contas acima, já vemos que só para armazenar os parâmetros, gradientes e estados do otimizador, um modelo LLM de 7 bilhões de parâmetros 
(modelo pequeno para os padrões atuais) pode exigir aproximadamente **100 GB de memória** (sem contabilizar o espaço ocupado pelos dados de entrada). 
Isso torna inviável treinar modelos dessa escala em GPUs de uso doméstico, como a RTX 4090, ou mesmo em placas profissionais topo de linha, como as Nvidia H100, 
que apesar de custarem centenas de milhares de reais, ainda não comportam toda a memória necessária para o treinamento de ponta a ponta.
Esse cenário ilustra o **alto custo e a complexidade envolvidos no treinamento de grandes modelos de linguagem**. Não se trata apenas do preço do hardware, 
mas também dos desafios de escalar infraestrutura, lidar com o consumo elevado de energia e otimizar cada etapa do processo para que o treinamento seja viável. 
Por isso, a maioria dos grandes LLMs só é treinada por empresas e laboratórios com recursos computacionais massivos e especializados.

> ⚠️ **Atenção:**  
> Os cálculos acima consideram apenas a memória exigida para os parâmetros e estados do modelo. Durante o treinamento real, o consumo total de memória pode ser ainda maior, 
já que precisamos incluir o armazenamento de dados, batches em processamento e outras estruturas temporárias.  
> Essa limitação de hardware é um dos principais motivos para o avanço de técnicas de paralelização, 
uso de quantização e soluções distribuídas em clusters de data centers para viabilizar o desenvolvimento de LLMs cada vez maiores.

# Exemplo de Treinamento Simples com um Modelo Básico de LLM (Transformer)

É fundamental entender como os modelos de linguagem aprendem padrões a partir de dados. Nesta seção, 
apresentamos um exemplo didático de treinamento com um **modelo Transformer básico** — arquitetura que serve de base para LLMs modernos, como GPT.

A ideia aqui é mostrar, de forma simples e prática, como um modelo de linguagem pode ser treinado do zero para prever a próxima palavra em frases, 
utilizando um conjunto de dados pequeno e controlado. Com esse experimento, conseguimos observar de perto:
- Como os dados são preparados e convertidos em sequências para o modelo,
- Como funciona o treinamento de um Transformer minimalista,
- E como as previsões do modelo refletem fielmente os padrões e os vieses presentes nos dados de treino.

Esse tipo de exemplo é essencial para entender o "coração" do aprendizado dos LLMs, facilitando a compreensão dos desafios e comportamentos que veremos em modelos maiores.
"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Dataset sintético: frases curtas e repetidas, todas com a mesma estrutura ("eu gosto de X"),
onde X varia entre cinco opções de alimentos. Isso cria um conjunto de dados simples e controlado,
ideal para observar como o modelo aprende padrões sequenciais e como os dados influenciam suas previsões."""

sentences = [
                "eu gosto de pizza",
                "eu gosto de bolo",
                "eu gosto de pão",
                "eu gosto de sorvete",
                "eu gosto de pastel",
            ] * 100

# 2. Tokenização por palavra
words = ["<PAD>"] + sorted(list(set(" ".join(sentences).split())))
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)

"""Cria pares de treino do tipo (contexto, próxima palavra):
Para cada frase, gera exemplos onde o input é um contexto de tamanho variável
e o target é sempre a próxima palavra real da frase.
Dessa forma, o modelo aprende a prever o próximo token dado qualquer parte da sequência.data_X = []"""

data_X = []
data_y = []
max_len = max(len(s.split()) for s in sentences)

for s in sentences:
    tokens = [word2idx[w] for w in s.split()]
    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        context += [word2idx["<PAD>"]] * (max_len - 1 - len(context))
        data_X.append(context)
        data_y.append(target)

X = torch.tensor(data_X)
y = torch.tensor(data_y)

"""
Define um modelo Transformer 'minimalista':
- Embedding: transforma índices de palavras em vetores densos.
- TransformerEncoder: aplica camadas de autoatenção e feedforward (dois heads, duas camadas).
- O método forward passa o input pela embedding, depois pelo encoder,
  pega apenas a representação do último token do contexto e projeta para o espaço do vocabulário.
- Assim, o modelo prevê a próxima palavra com base no contexto fornecido.
"""

#Arquitetura da RN (Rede Neural)
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead) # MODELO TRANSFORMER
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x[-1]
        return self.fc(x)

model = MiniTransformer(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

"""Treinamento do modelo:
Para cada época, o modelo tenta adivinhar a próxima palavra de cada exemplo do treino.
Ajusta seus parâmetros para errar menos a cada rodada.
A cada 10 épocas, mostra o valor do erro (Nesse caso, quanto menor, melhor o aprendizado)."""

epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)      # Saída: (batch, vocab_size)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

"""Essa função começa com uma frase inicial ("seed") e gera novas palavras uma por uma.
A cada passo, mostra quais palavras o modelo acha mais prováveis para continuar a frase,
escolhe a mais provável e segue gerando até completar o número de passos definido."""

def generate_with_probs(seed, steps=3):
    seq = [word2idx[w] for w in seed.split()]
    print(f"Contexto inicial: \"{seed}\"")
    for step in range(steps):
        inp = seq + [word2idx["<PAD>"]] * (max_len - 1 - len(seq))
        inp = torch.tensor(inp).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        sorted_indices = probs.argsort()[::-1]
        print(f"\nPasso {step+1}:")
        print("Palavras já geradas:", " ".join(idx2word[i] for i in seq))
        print("Probabilidades para todas as palavras do vocabulário:")
        for idx in sorted_indices:
            print(f"  {idx2word[idx]:<10} -> {probs[idx]:.3f}")

        next_word = sorted_indices[0]
        print(f"Token escolhido: {idx2word[next_word]} (prob={probs[next_word]:.3f})")
        seq.append(next_word)
    print("\nTexto final:", " ".join(idx2word[i] for i in seq))
    return " ".join(idx2word[i] for i in seq)

generate_with_probs("eu", steps=3)

"""# Sobre Viés de Dados em Modelos de Linguagem

Neste experimento, treinamos um pequeno modelo Transformer apenas com frases repetidas do tipo:  
**"eu gosto de pizza", "eu gosto de bolo", "eu gosto de pão", "eu gosto de sorvete", "eu gosto de pastel".**

O modelo aprendeu a prever corretamente a sequência esperada, como podemos ver nos primeiros passos da geração:

- **Passo 1** (após "eu"):  
  O modelo prevê "gosto" com probabilidade alta. Isso acontece porque, nos dados de treino, "gosto" sempre vem logo após "eu". 
  O modelo não conhece nenhum outro contexto nesse ponto.

- **Passo 2** (após "eu gosto"):  
  Agora, "de" é a única palavra possível que o modelo viu depois desse contexto, então sua probabilidade dispara para. 
  Isso mostra como o modelo se torna confiante quando só conhece uma opção de sequência.

- **Passo 3** (após "eu gosto de"):  
  Neste ponto, o modelo precisa escolher entre as diferentes comidas. Como todas aparecem exatamente com a mesma frequência e posição nos dados de treino, 
  as probabilidades ficam **muito balanceadas**. O modelo fica “indeciso” porque, estatisticamente, qualquer uma dessas opções é igualmente provável 
  — isso reflete perfeitamente o viés do dataset.

## O que aprendemos?

Esse resultado demonstra na prática como **modelos de linguagem são completamente dependentes dos dados em que são treinados**.  
- Se os dados apresentam o mesmo padrão, o modelo ficará enviesado, prevendo exatamente essas sequências.
**Por isso, ao treinar modelos maiores para tarefas reais, é essencial garantir dados diversos, bem distribuídos e representativos do uso que se espera 
— caso contrário, o modelo vai apenas reproduzir (e amplificar) os vieses dos dados.**
Com modelos maiores e conjuntos de dados muito mais extensos, a situação fica ainda mais complexa. Modelos de linguagem de grande porte conseguem aprender 
**muitos padrões diferentes ao mesmo tempo**, inclusive padrões extremamente sutis e relações que, muitas vezes, são invisíveis para os humanos.
Isso significa que, quanto mais dados fornecemos para o treinamento, mais “nuances” e características escondidas dos textos o modelo é capaz de absorver 
— inclusive tendências, repetições, vieses culturais ou de opinião que aparecem apenas em certos contextos ou grupos.
Esses padrões podem ser tão discretos que passam despercebidos até mesmo para quem prepara ou revisa os dados, principalmente em conjunto de dados extensos. 
Por isso, o viés em modelos grandes é ainda mais delicado: **o modelo pode aprender e reproduzir associações não óbvias, que só surgem ao analisar bilhões de exemplos juntos**.
---

Antes de explorarmos as nuances da geração de texto com diferentes parâmetros, vale lembrar que os grandes modelos de linguagem apresentam diversos desafios práticos e éticos. 
Além do viés nos dados, modelos dessa escala podem gerar informações falsas (“alucinações”), reforçar estereótipos sociais, apresentar linguagem tóxica, 
comprometer a privacidade dos dados utilizados em seu treinamento e exigir um enorme custo computacional e energético para seu desenvolvimento e uso.
Esses desafios não são exclusivos dos modelos gigantescos: mesmo modelos menores, quando expostos a conjuntos de dados limitados ou enviesados, 
tendem a repetir e amplificar padrões presentes nesses dados. Por isso, além de analisar a arquitetura e o desempenho dos LLMs, 
é fundamental compreender como suas decisões probabilísticas e os parâmetros usados na geração de texto impactam as respostas produzidas.

Entre esses parâmetros, a **temperatura** se destaca como um dos mais relevantes. Ela controla o equilíbrio entre previsibilidade e criatividade durante a geração, 
afetando diretamente como o modelo utiliza o conhecimento aprendido nos dados de treino. Ao experimentar com diferentes valores de temperatura, 
podemos observar na prática como o modelo navega entre respostas mais conservadoras e outras mais criativas ou inesperadas.

A seguir, vamos explorar, de forma prática, o efeito da temperatura na geração de texto de um modelo de linguagem.

---
## Explorando o Efeito da Temperatura na Geração de Texto

Uma coisa que conseguimos perceber até aqui é que os modelos de linguagem são probabilísticos, mas conseguimos controlar o quão probabilisticos eles são. Para isso vamos utilizar o parâmetro de temperatura
A temperatura é um dos principais parâmetros que influenciam o comportamento de um modelo de linguagem durante a geração de texto. Ela controla o grau de criatividade e aleatoriedade nas respostas do modelo.

Neste experimento, vamos utilizar um modelo de linguagem pequeno e atual para gerar diferentes continuações para o mesmo prompt, variando apenas o valor da temperatura. Assim, será possível observar como a escolha desse parâmetro afeta o tipo de texto produzido:  
- **Temperaturas mais baixas** (próximas de 0) fazem o modelo ser mais conservador, escolhendo quase sempre as palavras mais prováveis.
- **Temperaturas médias** promovem equilíbrio entre criatividade e coerência.
- **Temperaturas altas** tornam as respostas mais imprevisíveis, podendo gerar textos criativos — mas também menos coerentes.

Essa exploração prática ajuda a entender como pequenos ajustes em parâmetros de geração podem impactar significativamente o resultado final de modelos de linguagem.

Abaixo, apresentamos um exemplo prático variando a temperatura na geração de texto:
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "Qwen/Qwen3-1.7B"

# Carregando tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Identificando dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando dispositivo: {device}")

# Carregando modelo
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto"
).to(device)

model.eval()

# Prompt
messages = [
    {
        "role": "user",
        "content": "Explique em português brasileiro por que Pelé é considerado o rei do futebol. Escreva um texto coerente e objetivo."
    }
]

# Aplicando template do Qwen3
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

# Tokenizando entrada
inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(device)

# Temperaturas
temperaturas = [0.2, 0.7, 1.0, 1.5]

for temp in temperaturas:
    print(f"\nTemperatura: {temp}")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1
        )

    # Recuperando somente os tokens novos
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]

    resposta = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
    print(resposta)

"""## Comentário sobre o experimento de temperatura

Ao variar o parâmetro de temperatura, é possível perceber claramente como a saída do modelo de linguagem muda:

- **Temperaturas baixas (0.2, 0.7):** O texto gerado é mais conservador, lógico e repetitivo, geralmente reproduzindo respostas seguras e padrões comuns que o modelo viu no treinamento. O modelo prefere sequências de palavras mais prováveis, com menor criatividade.
- **Temperatura intermediária (1.0):** O modelo passa a apresentar mais variedade, trazendo ideias novas e saídas menos previsíveis, mas ainda mantém certa coerência.
- **Temperaturas altas (1.5, 2.5):** A criatividade aumenta bastante, mas a coerência das respostas diminui. O texto pode ficar confuso, desconexo ou até mesmo conter frases inesperadas e sem sentido.

Esse experimento mostra como a temperatura funciona como um verdadeiro “controle de criatividade” do modelo: quanto maior o valor, mais diversidade e imprevisibilidade na resposta, mas também maior o risco de incoerências ou alucinações.

## Conclusão

Neste notebook vimos como modelos são construídos, treinados e utilizados, além dos principais desafios envolvidos, como viés, alucinação e demanda por grandes recursos computacionais.

Através de exemplos práticos, experimentamos a geração de texto com diferentes modelos, treinamos um mini-transformer e observamos como o comportamento do modelo depende diretamente dos dados de treino. Também analisamos como parâmetros como a temperatura influenciam a criatividade e a coerência das respostas.
"""