import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s -  %(levelname)s:  %(message)s')
import os, time, numpy as np, pandas as pd, shutil, random, tensorflow as tf
import matplotlib.pyplot as plt

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
    trueMatch = 0 # predict = authorize, real = authorize
    trueReject = 0 # predict = reject, real = reject
    falseMatch = 0 # predict = authorize, real = reject
    falseReject = 0 # predict = reject, real = authorize
    numTest = 0
    listRegistered = []
    listFetched = []
    registered=True

    recognizedir = 'data/test/recognize'
    for user in os.listdir(recognizedir):
        numTest+=1
        nameRegistered = user.split("_")[0]
        if nameRegistered[:3] == "CFD" or '-' in nameRegistered: # Check whether the given user is registered in DB
            listRegistered.append(None)
            registered = False
        else:
            listRegistered.append(nameRegistered)
            registered=True

        fp = os.path.join(recognizedir, user)

        try:
            user_vector = get_vector(fp, model, detect_faces)
            data = find_closest(pgcon, user_vector.flatten(), threshold=threshold)
            data = data.get('data', None)

            if (data == None or len(data) == 0) and not registered:
                listFetched.append(None)
                trueReject += 1
            else:
                nameFetched=data[0][0]
                listFetched.append(nameFetched)

                if nameRegistered == nameFetched:
                    trueMatch+=1
                elif registered == False: # real: reject, predict: match
                    falseMatch+=1
                else:
                    falseReject+=1 # real: match, predict: reject
        except:
            listFetched.append(None)
            trueReject+=1

    summaryDF = pd.DataFrame({'registered_as': listRegistered,
                              'fetched_as': listFetched, 
                              'threshold': [threshold for _ in range(len(listFetched))]
                              })
    summaryDF.to_csv(f'./test_result/fr_performance_{threshold}.csv')

    return numTest, trueMatch, trueReject, falseMatch, falseReject

def threshold_benchmarking(pgcon, model, detect_faces:function, thres_range):
    """
        Overall test biometric matching engine performance evaluation for in range decision threshold

        Output: FAR (False Acceptance Rate), FRR (False Rejection Rate)
    """
    tp = []
    tn = []
    fp = []
    fn = []
    thresList = []

    for i, thres in enumerate(thres_range):
        numTest, trueMatch, trueReject, falseMatch, falseReject = fr_performance(pgcon, threshold=thres, model=model, detect_faces=detect_faces)
        logging.info(f"""
        Test #{i}:
        
        Decision Threshold: {thres}
        {trueMatch}/{numTest} recognized from the overall test with the decision threshold: {thres}\n
        Number of True Match: {trueMatch}
        Number of True Reject: {trueReject}

        Number of False Match: {falseMatch}
        Number of False Reject: {falseReject}
        """)

        thresList.append(thres)
        tp.append(trueMatch)
        tn.append(trueReject)
        fp.append(falseMatch)
        fn.append(falseReject)


    df = pd.DataFrame({
        'threshold': thresList, 
        'trueMatch': tp,
        'trueReject': tn,
        'falseMatch': fp,
        'falseReject': fn
    })

    df['FAR'] = df['falseMatch']/(df['falseMatch'] + df['trueReject'])
    df['FRR'] = df['falseReject']/(df['falseReject']+df['trueMatch'])
    df.to_csv('./test_result/threshold_benchmarking.csv')
    plt.plot(df['FAR'], df['FRR'])
    plt.savefig('./threshold_benchmarking.png')

if __name__ == '__main__':
    # STARTS HERE
    # Specify the path to your model here (tensorflow only) ...
    model_path = 'pre-trained/vggface2.h5'
    # For non-tensorflow model, you need to override the function (Specify your own function)
    model = load_model(model_path)

    # CREATE PG_VECTOR DATABASE FOR STORING BIOMETRIC TEMPLATE (FACE EMBEDDING) with the DDL query below:
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

    # Override necessary method here...
    # def detect_faces(img: np.ndarray) -> tuple[int,int,int,int]:
    #     """
    #     Detect face from the given image parameter. Returning coordinates x,y,w,h
    #     """
    #     # Do face detection

    #     return x, y, w, h

    # Check FR model performance on the registered biometric template
    threshold_benchmarking(pgcon=pgconnection, model=model, detect_faces=detect_faces, thres_range=np.arange(0.5, 1, 0.01))
