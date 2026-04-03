from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langsmith import Client as LangSmithClient
from dotenv import load_dotenv
load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Use esta ferramenta para resolver expressões matemáticas simples."""
    try:
        result = str(eval(expression))
    except Exception as e:
        result = f"Erro ao avaliar a expressão: {e}"

    return result


@tool
def web_search(query: str) -> str:
    """Use esta ferramenta para buscar a capital de um país."""

    capitais_mundiais = {
        "Brasil": "Brasília",
        "Estados Unidos": "Washington D.C.",
        "Reino Unido": "Londres",
        "França": "Paris",
        "Alemanha": "Berlim",
        "Japão": "Tóquio",
        "China": "Pequim",
        "Rússia": "Moscou",
        "Canadá": "Ottawa",
        "Austrália": "Canberra",
        "Argentina": "Buenos Aires",
        "México": "Cidade do México",
        "Itália": "Roma",
        "Espanha": "Madri",
        "Portugal": "Lisboa",
        "Egito": "Cairo",
        "África do Sul": "Pretória",
        "Índia": "Nova Délhi",
        "Coreia do Sul": "Seul",
        "Colômbia": "Bogotá",
    }

    for pais, capital in capitais_mundiais.items():
        if pais.lower() in query.lower():
            return f"A capital de {pais} é {capital}."

    return f"Não foi possível localizar informações sobre '{query}' na base de dados."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, web_search]

tools_description = "\n".join([f"{t.name}: {t.description}" for t in tools])
tool_names = ", ".join([t.name for t in tools])

# Baixa o prompt ReAct do LangChain Hub em vez de definir manualmente
prompt = LangSmithClient().pull_prompt("hwchase17/react")
system_prompt = prompt.format(
    tools=tools_description,
    tool_names=tool_names,
    input="",
    agent_scratchpad="",
).strip()

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# Exemplo de uso
response = agent.invoke({"messages": [("human", "Qual é a capital do Romenia?")]})
print(response["messages"][-1].content)

# Exemplo de uso com uma questão que não pode ser respondida pelas ferramentas
response = agent.invoke({"messages": [("human", "E qual é a raiz quadrada de 16?")]})
print(response["messages"][-1].content)

# Exemplo de uso com uma questão que não pode ser respondida pelas ferramentas
response = agent.invoke({"messages": [("human", "O que é uma baleia?")]})
print(response["messages"][-1].content)
