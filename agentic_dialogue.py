from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
load_dotenv()

class Message:
    def __init__(self, speaker: str, receiver: str, content: str, responseObject: ChatCompletion | None):
        self.speaker = speaker
        self.receiver = receiver
        self.content = content
        self.responseObject = responseObject # store the response object for later analysis
    
    def to_dict(self):
        return {
            "speaker": self.speaker,
            "receiver": self.receiver,
            "content": self.content,
            "responseObject": None if self.responseObject is None else self.responseObject.model_dump()
        }
    
    def __str__(self):
        return f"{self.speaker} -> {self.receiver}: {self.content}"

class Conversation:
    def __init__(self):
        self.history: list[Message] = []
    
    def add_message(self, message: Message):
        self.history.append(message)
    
    def print(self):
        for message in self.history:
            print(f"\033[93m{message.speaker} --> {message.receiver}\033[0m: {message.content}")
    
    def to_list(self):
        return [message.to_dict() for message in self.history]

class Agent:
    def __init__(self, name, system_message, model_name):
        self.model_name = model_name
        self.client = OpenAI()
        self.name = name
        self.system_message = system_message
        self.messages = [
            {"role": "system", "content": self.system_message},
        ]
        self.raw_responses = []
    
    def _generate_response(self):
        '''Produces a response to the message history and returns the response.'''
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.raw_responses.append(response)
        return response
    
    def respond(self, asking_agent: 'Agent'):
        '''Generates a response and appends it to both agents' message histories.'''
        response = self._generate_response()
        reply = response.choices[0].message.content
        # print(f"{self.name} --> {asking_agent.name}: {reply}")
        self.messages.append({"role": "assistant", "content": reply})
        asking_agent.messages.append({"role": "user", "content": reply})
        return response

    def ask(self, other: 'Agent', message: str):
        '''Asks another agent a question and appends it to both agents' message histories.'''
        # print(f"{self.name} --> {other.name}: {message}")
        self.messages.append({"role": "assistant", "content": message})
        other.messages.append({"role": "user", "content": message})
    
    def initiate_conversation(self, other: 'Agent', message: str, max_turns: int = 5):
        '''Initiates a conversation with another agent until a reply contains "TERMINATE" or max_turns is reached.'''
        terminate = lambda reply: 'TERMINATE' in reply
        choose_speaker = lambda current: other if current == self else self
        
        # To keep track of the conversation
        conversation = Conversation()
        
        self.ask(other, message)
        conversation.add_message(Message(self.name, other.name, message, None))
        
        response = other.respond(self)
        conversation.add_message(Message(other.name, self.name, response.choices[0].message.content, response))
        turns = 2
        current_speaker = other  # self asked, other responded
        
        while not terminate(response.choices[0].message.content) and turns < max_turns:
            next_speaker = choose_speaker(current_speaker)
            response = next_speaker.respond(current_speaker) # next speaker responds to the current speaker
            conversation.add_message(Message(next_speaker.name, current_speaker.name, response.choices[0].message.content, response))
            current_speaker = next_speaker
            turns += 1
        
        return conversation