import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s -  %(levelname)s:  %(message)s')
import os, time, numpy as np, pandas as pd, shutil, random, tensorflow as tf
from datetime import datetime

from recognize import get_vector, register_user, find_closest, detect_faces
from utils.db_conn import PostgrePy

def load_model(model_path:os.PathLike):
    """
        Need adjustment for non-Tensorflow model. Do method overload.

        model_path: Path to model checkpoint or model weights
    """
    model = tf.keras.models.load_model(model_path)

    return model

def register(pgcon, enroldir:os.PathLike):
    """
        Register user within a directory to the database.

        \npgcon: Postgres class API that handles postgres cursor
        \nenroldir: Images directory for user to be enrolled in the database
    """

    # Enrol user to the database
    for user in os.listdir(enroldir):
        name = user.split("_")[0]
        logging.info(f"Registering user {name} to DB")
        fp = os.path.join(enroldir, user)
        try:
            user_vector = get_vector(fp, model, detect_faces)
            register_user(pgcon, {'name': name}, user_vector.flatten())
            logging.info(f"{name} successfully registered")
            time.sleep(0.2)
        except Exception as error:
            logging.error(f"Failed to register user {name}\nDue to: {error}")

    pgcon.con.commit()
    pgcon.cur.close()
    pgcon.con.close()
    return

def fr_performance(pgcon, threshold:float, model, detect_faces:function):
    """
        Test biometric matching engine performance evaluation for a specific decision threshold
    """
    trueVerif = 0
    falseVerif = 0
    fta = 0
    numTest = 0
    listRegistered = []
    listFetched = []

    # Recognize dir
    recognizedir = 'data/recognize'

    for user in os.listdir(recognizedir):
        numTest+=1
        name = user.split("_")[0]
        listRegistered.append(name)
        fp = os.path.join(recognizedir, user)

        try:
            user_vector = get_vector(fp, model, detect_faces)
            data = find_closest(pgcon, user_vector.flatten(), threshold=threshold)
            data = data.get('data', None)

            if data == None or len(data) == 0:
                listFetched.append(None)
                fta += 1
            else:
                nameFetched=data[0][0]
                listFetched.append(nameFetched)
                if name == nameFetched:
                    trueVerif+=1
                else:
                    falseVerif+=1
        except:
            listFetched.append(None)
            fta+=1


    summaryDF = pd.DataFrame({'registered_as': listRegistered, 'fetched_as': listFetched, 'threshold': [threshold for _ in range(len(listFetched))]})
    summaryDF.to_csv(f'./test_result/fr_performance_{threshold}.csv')

    return numTest, trueVerif, falseVerif, fta

def threshold_benchmarking(pgcon, model, detect_faces:function, thres_range):
    """
        Overall test biometric matching engine performance evaluation for in range decision threshold

        Output: FAR (False Acceptance Rate), FRR (False Rejection Rate)
    """

    trueList = []
    falseList = []
    ftaList = []
    thresList = []

    for i, thres in enumerate(thres_range):
        numTest, trueVerif, falseVerif, fta = fr_performance(pgcon, threshold=thres, model=model, detect_faces=detect_faces)
        logging.info(f"""
        Test #{i}:
        
        Decision Threshold: {thres}
        {trueVerif}/{numTest} succeed from the overall test with the decision threshold: {thres}\n
        Number of match: {trueVerif}
        Number of non-match: {falseVerif}
        Number of FTA: {fta}

        False Acceptance Rate: {round(trueVerif/numTest, 3)}\nFalse Rejection Rate: {round(falseVerif/numTest, 3)}
        """)

        thresList.append(thres)
        trueList.append(trueVerif)
        falseList.append(falseVerif)
        ftaList.append(fta)


    pd.DataFrame({'threshold': thresList, 'match': trueList, 'reject': falseList, 'fta': ftaList}).to_csv('./test_result/threshold_benchmarking.csv')

if __name__ == '__main__':
    # STARTS HERE
    # Specify the path to your model here ...
    model_path = 'pre-trained/vggface2.h5'
    model = load_model(model_path)

    # CREATE DATABASE FOR STORING BIOMETRIC TEMPLATE (FACE EMBEDDING) with the DDL query below:
    # -- DROP TABLE public.users;

    # CREATE TABLE public.users (
    # 	id bigserial NOT NULL,
    # 	"name" varchar NULL,
    # 	embedding public.vector NULL,
    # 	registered_at timestamp NULL,
    # 	CONSTRAINT users_pkey PRIMARY KEY (id)
    # );

    # Create .env file completing the variables below:
    # DB_DATABASE=db_name
    # DB_USER=username
    # DB_PASS=password
    # DB_HOST=db_host_address
    # DB_PORT=db_port
    pgconnection = PostgrePy()

    # Specify users image directory path to be registered to the database ...
    enroldir = 'data/register'
    register(enroldir)

    # Check FR model performance on the registered biometric template
    threshold_benchmarking(pgcon=pgconnection, model=model, detect_faces=detect_faces, thres_range=np.arange(0.5, 1, 0.01))
