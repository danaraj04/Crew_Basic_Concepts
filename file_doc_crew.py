from dotenv import load_dotenv

load_dotenv()
#print("Setup done! CrewAI can now use your OpenAI Key")

from crewai import Agent, Task, Crew, LLM

llm = LLM(model="gpt-4.1-mini")
print("OpenAi brain is ready!")

text="""Our shop is open from 9 AM to 6 PM.
			We sell shoes, bags and belts.
			Returns are allowed within 7 days."""
with open("shopping_info.txt","w")as f:
    f.write(text)
print("shopping_info.txt created!")

from crewai_tools import FileReadTool

#create the tool
read_tool = FileReadTool(file_path="shopping_info.txt")

#from crewai import Agent
assistant = Agent(
			role="Shop Assistant",
			goal="Answer question using the shop file",
			backstory="You read the shop file and answer customers.",
            tools=[read_tool],
			llm=llm,
			)
print("Agent now has a file reading tools!") 

task = Task(
			description="Using the file. what time does the shop close?",
			expected_output="The closing time.",
			agent=assistant,
			)
crew = Crew(agents=[assistant], tasks=[task])
result = crew.kickoff()
print(result)