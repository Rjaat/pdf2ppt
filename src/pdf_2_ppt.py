# importing libraries
import os
import re
from unstructured.partition.pdf import partition_pdf
from langchain.vectorstores import FAISS, Chroma
# from langchain_community.vectorstores import Chroma, FAISS
from src.modelling import Modelling
from src.vectordb import VectorDB
from src.logging import logger

class RAG:
    def __init__(self) -> None:
        logger.info('======RAG Function Started======')
        # self.user_prompt = user_prompt

    def extract_slides(self, text):
        slide_pattern = r'\*\*Slide (\d+): (.+?)\*\*[\s\S]*?Points:(.+?)(?=\n\n\[|\Z)'
        slides = re.findall(slide_pattern, text, re.DOTALL)

        final_list = []
        for slide in slides:
            slide_number = slide[0].strip()
            title = slide[1].strip()
            
            slide_dict = {
                "Topic": title,
                "Summary": slide[2].strip()
            }

            final_list.append(slide_dict)

        return final_list

   
    def get_response(self, question, user_prompt):
        # your pdf path
        file_path = 'dataset/attention_all_you_need.pdf'

        logger.info('======Extracting Text, Images and Tables from PDF file======')
        # extracting Text, Images and Tables from PDF file
        raw_pdf_elements = partition_pdf(
                                filename = file_path,
                                extract_images_in_pdf=True,
                                infer_table_structure=True,
                                chunking_strategy="by_title",
                                max_characters=4000,
                                new_after_n_chars=3800,
                                combine_text_under_n_chars=2000,
                                extract_image_block_output_dir= "./images",
                              )
        if user_prompt == None:
            print('++++++ inside system prompt ++++++')
            prompt_template = """
                    You are a PPT creator AI expert your role is to generate atleast 8 detailed ppt slides contaning topic and five points based on the following context,
                    which can include text, images and tables.
                    slide formate should be slide: title: points:
                    {context}
                    Question: {question}
                    Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
                    Just return the helpful answer in as much as detailed possible.
                    Answer:
                """
        else:
            print('++++++ inside user prompt ++++++')
            prompt_template = user_prompt
            
        obj1 = VectorDB(raw_pdf_elements)
        if not os.path.exists('/content/vector_store'):
            vectorstore = obj1.generate_embeding()

        elif os.path.exists('/content/vector_store'):
            print('loading vector db from local')
            embedding = obj1.get_embeding()
            vectorstore = Chroma(persist_directory="./vector_store", embedding_function = embedding)

        obj2 = Modelling()
        qa_chain = obj2.get_model(prompt_template)

        relevant_docs = vectorstore.similarity_search(question, k=5)

        logger.info('======Generating PPT Content======')
        context = ""
        relevant_images = []
        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
            elif d.metadata['type'] == 'image':
                context += '[image]' + d.page_content
                relevant_images.append(d.metadata['original_content'])
        result = qa_chain.run({'context': context, 'question': question})
        
        resultp = self.extract_slides(result)
        return resultp, relevant_images



