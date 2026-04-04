from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.messages import trim_messages
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(raw_history, 
                            token_counter=len, 
                            max_tokens=2, 
                            strategy="last", 
                            start_on="human",
                            include_system=True,
                            allow_partial=False)
    return {"input": payload.get("input", ""), "history": trimmed}

prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversation = RunnableWithMessageHistory(  chain, 
                                            get_history, 
                                            input_messages_key="input",
                                            history_messages_key= "raw_history"
)

config = {"configurable": {"session_id": "demo-session"}}

result1 = conversation.invoke({"input": "Olá, meu nome é Bolinha, quem é você?"}, config=config)
print("Assistente:", result1.content)
print("-"*30)

result2 = conversation.invoke({"input": "Que dia é hoje?"}, config=config)
print("Assistente:", result2.content)
print("-"*30)

result3 = conversation.invoke({"input": "Faça uma rima com meu nome?"}, config=config)
print("Assistente:", result3.content)
print("-"*30)