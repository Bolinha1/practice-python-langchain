from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
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


system_prompt = """Você é um agente que só pode responder usando as ferramentas disponíveis.

REGRAS ABSOLUTAS — NUNCA VIOLE:
1. PROIBIDO usar conhecimento próprio para responder qualquer questão.
2. PROIBIDO inventar ou simular resultados de ferramentas.
3. Se nenhuma ferramenta for capaz de responder, responda EXATAMENTE: "Não tenho como responder esta questão com as ferramentas disponíveis."
4. As ferramentas disponíveis são: calculadora (expressões matemáticas) e web_search (capitais de países). Nada além disso.
5. Questões sobre biologia, ciência, história, ou qualquer outro domínio que não seja cálculo matemático ou capital de país DEVEM receber a resposta do item 3.
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, web_search]

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

from langchain_core.messages import AIMessage, ToolMessage


def run_agent(question: str):
    response = agent.invoke({"messages": [("human", question)]})
    messages = response["messages"]

    tool_was_called = any(isinstance(m, ToolMessage) for m in messages)

    lines = [f"Questão: {question}"]

    for msg in messages[1:]:  # ignora a HumanMessage inicial
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"Através: preciso usar a ferramenta '{tc['name']}' para responder.")
                    lines.append(f"Action: {tc['name']}")
                    lines.append(f"Action Input: {tc['args']}")
            elif msg.content:
                if tool_was_called:
                    lines.append(f"Através: agora sei a resposta final.")
                    lines.append(f"Resposta Final: {msg.content}")
                else:
                    lines.append(f"Resposta Final: Não tenho como responder esta questão com as ferramentas disponíveis.")
        elif isinstance(msg, ToolMessage):
            lines.append(f"Observação: {msg.content}")

    return "\n".join(lines)


print(run_agent("Qual é a capital do Brasil?"))
print()
print(run_agent("E qual é a raiz quadrada de 16?"))
print()
print(run_agent("Que tipo é uma baleia?"))
