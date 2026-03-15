from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system = ("system", "voce é um assisten que responde perguntas no {style} estilo")

user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="funny", question="Quem é Alan Turing")


for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0.5)
result = model.invoke(messages)
print(result.content)
