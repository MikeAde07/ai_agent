
#import necessary modules and packages
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
#AgentExecutor is a way to execut the agent
from langchain.agents import create_tool_calling_agent, AgentExecutor
#to use our tool we must import it
from tools import search_tool, wiki_tool, save_tool

# loads environment variable file (.env)
load_dotenv()

#inherit from BaseModel from pydantic
class ResearchResponse(BaseModel) :
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str] 
   
#generate our LLM
llm = ChatOpenAI(model="gpt-4o-mini")
#allows us to take the output of the LLM and parse it into the ResearchResponse model and use it like a normal object inside of our code
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#create our prompt with the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            #information to the LLM so it knows what it's supposed to be doing
            "system", 
            """
            You are a reasearch assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools =[search_tool, wiki_tool, save_tool]
#create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
 
#create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#query is a user input
query = input("What can I help you research?")
raw_response = agent_executor.invoke({"query": query})


try :
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response) 
except Exception as e :
    print("Error parsing response", e, "Raw Response - ", raw_response)
