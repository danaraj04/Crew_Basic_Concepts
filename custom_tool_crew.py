from dotenv import load_dotenv

load_dotenv()
#print("Setup done! CrewAI can now use your OpenAI Key")

from crewai.tools import tool

@tool("Add Tool")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together andreturn the result."""
    return a + b

print("Custom tool created!")


from crewai import Agent, Task, Crew, LLM

llm = LLM(model="gpt-4.1-mini")
#print("OpenAi brain is ready!")

#from crewai import Agent
math_helper = Agent( role="Maths Helper",
				goal="solve simple math using tools",
				backstory="You always use yor tools to calculate.",
                tools=[add_numbers],
				llm=llm,
				)

print("Agent now has a Custom tools!") 