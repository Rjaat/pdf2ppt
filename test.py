from src.pdf_2_ppt import RAG
from src.logging import logger

logger.info('\n===== logger start======\n')
obj = RAG()
result, relevant_images = obj.get_response("attention all ypu need")
print('PPT \n', result)


logger.info('logger ender')