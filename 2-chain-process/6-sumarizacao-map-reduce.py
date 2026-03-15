from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """A LangChain é uma biblioteca de código aberto para desenvolvimento de aplicações de IA. Ela fornece uma estrutura para criar e gerenciar cadeias de processamento de linguagem natural, permitindo que os desenvolvedores construam sistemas complexos de IA de forma modular e eficiente. A biblioteca suporta integração com diversos modelos de linguagem, facilitando a criação de soluções personalizadas para uma variedade de casos de uso, como chatbots, assistentes virtuais e análise de texto."""


splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

parts = splitter.create_documents([long_text])

for part in parts:
    print(part.page_content)
    print("-"*30)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain_sumarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

result = chain_sumarize.invoke({"input_documents": parts})

print(result)