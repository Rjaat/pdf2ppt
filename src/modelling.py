from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from src.logging import logger

class Modelling:

    def __init__(self) -> None:
        print('modelling constructor')
        
    def get_model(self, prompt):
        logger.info('======Loading Model======')
        
        chain = LLMChain(
            llm = Ollama(model="llama3"),
            prompt=PromptTemplate.from_template(prompt)
        )
        logger.info('======Model Loaded Successfully======')
        return chain

    def get_embeding(self):
        logger.info('======Loading Embedding======')
        # openai_api_key = userdata.get('OPENAI_API_KEY')
        # return OpenAIEmbeddings(openai_api_key=openai_api_key)
        embedding = OllamaEmbeddings(model="llama3")
        logger.info('======Embedding Loaded Successfully======')
        return embedding

    def get_model_image(self, img_path):
        logger.info('======Loading Image Model======')
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        img_txt = captioner(img_path)
        logger.info('======Image Model Loaded Successfully======')
        return img_txt[0]['generated_text']
