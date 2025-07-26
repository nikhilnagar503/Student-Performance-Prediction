import logging
import os
import datetime

logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',     
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    logging.info("logging has started")
    

