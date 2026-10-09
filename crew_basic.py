from dotenv import load_dotenv

load_dotenv()
print("Setup done! CrewAI can now use your OpenAI Key")

from crewai import Agent, Task, Crew
print("CrewAI is ready to use.") 

from crewai import LLM
llm = LLM(model="gpt-4.1-mini")
print("OpenAi brain is ready!")

#from crewai import Agent
helper = Agent( role="Helper",
				goal="Answer question simply",
				backstory="You explain things to Beginner",
				llm=llm,
				)
print("Agent created with the OpenAi brain!")
