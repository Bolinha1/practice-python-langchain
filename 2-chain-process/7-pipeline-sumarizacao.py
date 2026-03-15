from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

long_text = """A LangChain é uma biblioteca de código aberto para desenvolvimento de aplicações de IA. Ela fornece uma estrutura para criar e gerenciar cadeias de processamento de linguagem natural, permitindo que os desenvolvedores construam sistemas complexos de IA de forma modular e eficiente. A biblioteca suporta integração com diversos modelos de linguagem, facilitando a criação de soluções personalizadas para uma variedade de casos de uso, como chatbots, assistentes virtuais e análise de texto."""


print("\n=== ETAPA 1: Dividindo o texto em partes (chunks) ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

parts = splitter.create_documents([long_text])

print(f"Texto dividido em {len(parts)} partes:\n")
for i, part in enumerate(parts, 1):
    print(f"[Parte {i}] {part.page_content}")
    print("-"*30)


print("\n=== ETAPA 2: Configurando o modelo e os prompts ===")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Modelo: gpt-4o-mini")

map_prompt = PromptTemplate.from_template("Faça um resumo conciso do seguinte texto. :\n ```{text}```")
map_chain = map_prompt | llm | StrOutputParser()
print("Map prompt: resume cada parte individualmente")

prepare_map_inputs = RunnableLambda(lambda docs:[{"text": doc.page_content} for doc in docs])
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt = PromptTemplate.from_template("Dado os seguintes resumos, gere um resumo sumarizado:\n {summaries}")
reduce_chain = reduce_prompt | llm | StrOutputParser()
print("Reduce prompt: combina todos os resumos em um único")

prepare_reduce_inputs = RunnableLambda(lambda summaries: "\n".join(summaries))
pipeline = map_stage | prepare_reduce_inputs | reduce_chain

print("\n=== ETAPA 3: Executando o pipeline (Map → Reduce) ===")
print("Enviando cada parte ao LLM para resumo individual (map)...")
print("Em seguida, combinando os resumos em um resumo final (reduce)...\n")

result = pipeline.invoke(parts)

print("=== RESULTADO FINAL ===")
print(result)