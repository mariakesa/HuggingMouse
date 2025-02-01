from smolagents import CodeAgent, HfApiModel
from mousetool import allen_engine
import os
import dotenv

dotenv.load_dotenv()

agent = CodeAgent(
    tools=[allen_engine],
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", token=os.getenv("HF_TOKEN"))
)
agent.run("What are the neuron id's in the Allen Brain Observatory API in the session 831882777?")