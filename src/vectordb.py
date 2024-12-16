
import uuid
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS, Chroma

from src.helper import Helper
from src.summary import Summary
from src.modelling import Modelling
from src.logging import logger

class VectorDB:
    def __init__(self, raw_pdf_elements) -> None:
        logger.info('======Vector-DB Function Started======')
        self.raw_pdf_elements =  raw_pdf_elements

    def generate_embeding(self):
        logger.info('======Generating Embeddings======')
        # creating object for summary
        obj1 = Summary(self.raw_pdf_elements)

        # calling get_text_summary function
        text_elements, text_summaries = obj1.get_text_summary()

        # calling get_image_summary function
        image_elements, image_summaries = obj1.get_image_summary()

        # calling get_text_summary function
        table_elements, table_structure, table_summaries = obj1.get_table_summary()
        obj1.get_tables(table_structure)

        # Create Documents and Vectorstore
        documents = []
        retrieve_contents = []

        for e, s in zip(text_elements, text_summaries):
            i = str(uuid.uuid4())
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'text',
                    'original_content': e
                }
            )
            retrieve_contents.append((i, e))
            documents.append(doc)

        for e, s in zip(table_elements, table_summaries):
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'table',
                    'original_content': e
                }
            )
            retrieve_contents.append((i, e))
            documents.append(doc)

        for e, s in zip(image_elements, image_summaries):
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'image',
                    'original_content': e
                }
            )
            retrieve_contents.append((i, s))
            documents.append(doc)

        obj3 = Modelling()
        embedding = obj3.get_embeding()
        vectorstore = Chroma.from_documents(documents=documents, embedding = embedding , persist_directory="./vector_store/")
        logger.info('======Embeddings Stored Successfully in Vector-DB======')
        return vectorstore