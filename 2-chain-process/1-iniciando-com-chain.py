from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


question_template = PromptTemplate(
    input_variables = ["name"], 
    template="Hi, I,m {name}! Conte uma piada com meu nome!"
)

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0.5)

chain = question_template | model 

result = chain.invoke({"name": "Bolinha"})

print(result.content)



