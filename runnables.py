from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)