from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()




@Tool(name="Calculadora", description="Use esta ferramenta para resolver expressões matemáticas simples.")
def calculator(expression: str) -> str:
    """Função simples para avaliar expressões matemáticas e retornar o resultado."""
    try:
        result = str(eval(expression))
    except Exception as e:
        result = f"Erro ao avaliar a expressão: {e}"
    
    return result


@Tool(name="web_search", description="Voce é uma ferramenta de busca na web.")
def web_search(query: str) -> str:
    """Função simulada para realizar uma busca na web e retornar um resultado fictício."""

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

prompt = PromptTemplate.from_template("""Responda as questões da melhor maneira possível.
Você tem acesso às seguintes ferramentas: {tools}

Utilize o seguinte formato ao raciocinar:

Questão: a questão de entrada que você deve responder
Através: seu raciocínio sobre o que fazer
Action: a ação a ser tomada, deve ser uma de [{tool_names}]
Action Input: a entrada para a ação
Observação: o resultado da ação
... (este ciclo de Através/Action/Observação pode se repetir N vezes)
Através: agora sei a resposta final
Resposta Final: a resposta final para a questão original

Regras:
- Se você escolher uma Action, NÃO inclua a Resposta Final no mesmo passo.
- Após Action e Action Input, pare e aguarde a Observação.
- Nunca pesquise na internet. Use apenas as ferramentas fornecidas.

Comece!

Questão: {input}
Através: {agent_scratchpad}""")
