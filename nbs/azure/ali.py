
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

# print(f"model: {os.environ['AZURE_OPENAI_35_TURBO_MODEL']}, \
# version: {os.environ['AZURE_OPENAI_35_TURBO_MODEL_VERSION']}, \
# deployment: {os.environ['AZURE_OPENAI_35_TURBO_DEPLOYMENT_ID']}, \
# endpoint: {os.environ['AZURE_OPENAI_API_BASE']}, \
# api key: {os.environ['AZURE_OPENAI_API_KEY']}")

# print(f"model: {os.environ['AZURE_OPENAI_4_TURBO_MODEL']}, \
# version: {os.environ['AZURE_OPENAI_4_TURBO_MODEL_VERSION']}, \
# deployment: {os.environ['AZURE_OPENAI_4_TURBO_DEPLOYMENT_ID']}, \
# endpoint: {os.environ['AZURE_OPENAI_API_BASE']}, \
# api key: {os.environ['AZURE_OPENAI_API_KEY']}")
# 



model=os.getenv("AZURE_OPENAI_35_16k_MODEL")
openai_api_version=os.getenv("AZURE_OPENAI_35_16k_MODEL_VERSION")
openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE")
deployment_name=os.getenv("DEPLOYMENT_ID")

print(f"Model[{model}]") 
print(f"openai_api_version [{openai_api_version}]") 
print(f"openai_api_key[{openai_api_key}]") 
print(f"azure_endpoint[{azure_endpoint}]") 
print(f"deployment_name[{deployment_name}]")



azureChatClient = AzureChatOpenAI(
    temperature=0,
    model=os.getenv("AZURE_OPENAI_35_16k_MODEL"),
    openai_api_version=os.getenv("AZURE_OPENAI_35_16k_MODEL_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    deployment_name=os.getenv("DEPLOYMENT_ID"),
)
TASK_DECONSTRUCTOR_BRAINSTORM = """
    Step1 :
     
    I have a problem related to {input}. Brainstorm three distinct solutions. Consider a variety of factors such as {perfect_factors}
    A:
    """

TASK_DECONSTRUCTOR_EVALUATION = """
    Step 2:
    
    For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors
    
    {solutions}
    
    A:"""

TASK_DECONSTRUCTOR_EXPANSION = """
    Step 3:

    For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled.

    {review}

    A:"""

TASK_DECONSTRUCTOR_RANK_SOLUTIONS = """
    Step 4:
    
    Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution
    {deepen_thought_process}
    
    A:"""





def brainstorm_phase():
    prompt = PromptTemplate(
        input_variables=["input", "perfect_factors"],
        template=TASK_DECONSTRUCTOR_BRAINSTORM
    )

    chain1 = LLMChain(
        llm=azureChatClient,
        prompt=prompt,
        output_key="ranked_solutions",
        verbose=True
    )
    return chain1


def evaluation_phase():
    prompt = PromptTemplate(
        input_variables=["solutions"],
        template=TASK_DECONSTRUCTOR_EVALUATION
    )

    chain2 = LLMChain(
        llm=azureChatClient,
        prompt=prompt,
        output_key="review",
        verbose=True
    )
    return chain2


def expansion_phase():
    prompt = PromptTemplate(
        input_variables=["review"],
        template=TASK_DECONSTRUCTOR_EXPANSION
    )

    chain3 = LLMChain(
        llm=azureChatClient,
        prompt=prompt,
        output_key="deepen_thought_process",
        verbose=True
    )
    return chain3


def rank_solutions_phase():
    prompt = PromptTemplate(
        input_variables=["deepen_thought_process"],
        template=TASK_DECONSTRUCTOR_RANK_SOLUTIONS
    )

    chain4 = LLMChain(
        llm=azureChatClient,
        prompt=prompt,
        output_key="ranked_solutions",
        verbose=True
    )
    return chain4


#def orchestrate(query: str):
#    overall_chain = SequentialChain(
#        chains=[brainstorm_phase(), evaluation_phase(), expansion_phase(), rank_solutions_phase()],
#        input_variables=["input", "perfect_factors"],
#        output_variables=["ranked_solutions"],
#        verbose=True
#    )
def orchestrate(query: str):
    overall_chain = SequentialChain(
        chains=[brainstorm_phase()],
        input_variables=["input", "perfect_factors"],
        output_variables=["ranked_solutions"],
        verbose=True
    )

    result = overall_chain({"input": query,
                            "perfect_factors": "you can contact your local embassy or apply for an emergency travel document to get home.  You can contact a local medical facility and report the incident to the police for the mugging."})

    print(result)

    # print(overall_chain({"input": "I've a UK citizen and lost my passport in bangkok and have been mugged",
    #                      "perfect_factors": "you can contact your local embassy or apply for an emergency travel document to get home.  You can contact a local medical facility and report the incident to the police for the mugging."}))
    return result


orchestrate("please help me, I've a UK citizen and lost my passport in bangkok and have been mugged")
