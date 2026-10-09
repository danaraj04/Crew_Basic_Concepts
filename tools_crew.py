from dotenv import load_dotenv

load_dotenv()
#print("Setup done! CrewAI can now use your OpenAI Key")

from crewai import Agent, LLM

llm = LLM(model="gpt-4.1-mini")
print("OpenAi brain is ready!")

from crewai_tools import FileReadTool

#create the tool
read_tool = FileReadTool()

#from crewai import Agent
helper = Agent( role="Helper",
				goal="Answer question simply",
				backstory="You explain things to Beginner",
                tools=[read_tool],
				llm=llm,
				)
print("Agent now has a file reading tools!") 