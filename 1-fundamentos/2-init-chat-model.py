from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

anwser_gemini = gemini.invoke("Hellow world")

print(anwser_gemini.content)

