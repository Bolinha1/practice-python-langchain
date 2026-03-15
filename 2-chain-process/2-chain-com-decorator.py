from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

@chain
def square(input:dict) -> dict:
    x = input["x"]
    return {"square_result": x * x}


question_template = PromptTemplate(
    input_variables = ["name"], 
    template="Hi, I,m {name}! Conte uma piada com meu nome!"
)

question_template2 = PromptTemplate(
    input_variables = ["square_result"], 
    template="Me fale sobre o número {square_result}!"
)

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0.5)

chain = question_template | model 

chain2 = square | question_template2 | model 


result = chain2.invoke({"x":10})

print(result.content)



