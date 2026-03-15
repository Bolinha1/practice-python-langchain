from langchain_core.runnables import RunnableLambda


def parseNumber(text:str) -> int:
    return int(text.strip())

parseRunnable = RunnableLambda(parseNumber)

number = parseRunnable.invoke("10")