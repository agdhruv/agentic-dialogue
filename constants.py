import os
import dotenv

dotenv.load_dotenv()

# Azure OpenAI configuration for Autogen
config_list = [
    # {
    #     "api_type": "azure",
    #     "model": os.environ['AZURE_DEPLOYMENT_ID'],
    #     "api_key": os.environ['AZURE_OPENAI_API_KEY'],
    #     "base_url": os.environ['AZURE_OPENAI_API_URL'],
    #     "api_version": "2024-02-01"
    # }
    {
        "model": "gpt-4-0125-preview",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

SUMMARY_PROMPT_TRUE_FALSE = (
    "Output a JSON object with the following fields: "
    "{'disagreement': '<A boolean value indicating whether there was a disagreement between the agents>', "
    "'answer': '<The answer to the question (true, false, or no_consensus)>'} "
    "Do not output any other things."
)

SUMMARY_PROMPT_MULTIPLE_CHOICE = (
    "Output a JSON object with the following fields: "
    "{'disagreement': '<A boolean value indicating whether there was a disagreement between the agents>', "
    "'answer': '<The answer to the question (A, B, C, or D, or no_consensus)>'} "
    "Do not output any other things."
)

# "Select the option: A, B, C, or D. "
# "If no consensus was reached, output 'No consensus'. "
# "Do not output any other things."