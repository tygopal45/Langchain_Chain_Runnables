from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableBranch

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    input_variables=["input"],
    template="Write a detailed report about. \n {input}"
)
prompt2 = PromptTemplate(
    input_variables=["input"],
    template="Write a short summary report about. \n {input}"
)

report_gen_chain = RunnableSequence(prompt1, model1, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model1, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke("Russia Vs Ukraine war")

print(result)