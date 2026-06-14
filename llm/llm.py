from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class  LLM(BaseModel):
    def __int__(self,api_key:str,model:str,base_url:str,temperature=0.2):
        self.api_key=api_key
        self.model=api_key
        self.base_url=api_key
        self.temperature=temperature

    def  get_llm(self):
        return  ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_retries=3
        )