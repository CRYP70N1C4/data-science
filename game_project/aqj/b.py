import logging


logger = logging.getLogger('aqj')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('aqj2.log', encoding="UTF-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
try:
    print(10 % 0)
except Exception as ex:
    logger.info("get detail exception id = %s,ex = %s","aaa",ex)