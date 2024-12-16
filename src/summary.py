
import re
import os
import uuid
import base64
import pandas as pd
from typing import Any
from bs4 import BeautifulSoup
from src.modelling import Modelling
from src.helper import Helper
from src.logging import logger



# Get elements
class Summary:

    def __init__(self, raw_pdf_elements):
        logger.info('======Summary Function Started======')
        self.raw_pdf_elements =  raw_pdf_elements
        self.prompt = """
                        Summarize the following {element_type}:
                        {element}
                    """



    def get_text_summary(self):
        logger.info('======Text Summary Started======')

        text_elements = []
        text_summaries = []

        #calling get model function
        obj = Modelling()
        summary_chain = obj.get_model(self.prompt)

        for e in self.raw_pdf_elements:
            if 'CompositeElement' in repr(e):
                text_elements.append(e.text)
                summary = summary_chain.run({'element_type': 'text', 'element': e})
                text_summaries.append(summary)

        obj1 = Helper()
        text_elements = obj1.top_sentences(text_elements)
        text_summaries = obj1.top_sentences(text_summaries)

        return text_elements, text_summaries


    def get_encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')


    def get_image_summary(self):
        logger.info('======Image Summary Started======')
        image_elements = []
        image_summaries = []

        # calling modelling 
        obj = Modelling()
        for i in os.listdir("./images"):
            if i.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join('./images', i)
                encoded_image = self.get_encode_image(image_path)
                image_elements.append(encoded_image)
                summary = obj.get_model_image(image_path)
                image_summaries.append(summary)

        return image_elements, image_summaries
    

    def get_table_summary(self):
        logger.info('======Table Summary Started======')
        table_elements = []
        table_structure = []
        table_summaries = []

        #calling get model function
        obj = Modelling()
        summary_chain = obj.get_model(self.prompt)

        for e in self.raw_pdf_elements:
            if 'Table' in repr(e):
                table_elements.append(e.text)
                table_structure.append(e.metadata.text_as_html)
                summary = summary_chain.run({'element_type': 'table', 'element': e})
                table_summaries.append(summary)

        obj1 = Helper()
        table_elements = obj1.top_sentences(table_elements)
        table_summaries = obj1.top_sentences(table_summaries)

        return table_elements, table_structure, table_summaries


    #Extracting tables and saving into folder
    def get_tables(self, table_structure):
        for i in range(len(table_structure)):
            table = table_structure[i]

            # for getting the header from
            data = []
            list_header = []
            soup = BeautifulSoup(table,'html.parser')
            header = soup.find_all("table")[0].find("tr")

            for items in header:
                try:
                    list_header.append(items.get_text())
                except:
                    continue

            # for getting the data
            HTML_data = soup.find_all("table")[0].find_all("tr")[1:]

            for element in HTML_data:
                sub_data = []
                for sub_element in element:
                    try:
                        sub_data.append(sub_element.get_text())
                    except:
                        continue
                data.append(sub_data)

            # Storing the data into dataframe
            try:
                dataFrame = pd.DataFrame(data = data,  columns = list_header)
            except:
                dataFrame = pd.DataFrame(data = data)

            os.makedirs('tables', exist_ok=True)
            # Converting Pandas DataFrame
            dataFrame.to_csv(f'tables/table_{i}.csv')
        
        logger.info('======Tables Stored in tables Folder======')

    
    

