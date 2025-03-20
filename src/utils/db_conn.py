import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

class PostgrePy():
    load_dotenv()

    def __init__(self):
        database = os.environ.get('DB_DATABASE')
        username = os.environ.get('DB_USER')
        password = os.environ.get('DB_PASS')
        host = os.environ.get('DB_HOST')
        port = os.environ.get('DB_PORT')

        self.con = psycopg2.connect(dbname=database, user=username, password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.cur.execute('''set timezone to 'Asia/Jakarta';''')

    def get_cursor(self):
        return self.cur

    def query(self, sql):
        self.cur.execute(sql)
        return {'header': [i[0] for i in self.cur.description], 'data': self.cur.fetchall()}

    def query_df(self, sql):
        self.cur.execute(sql)
        data = {'header': [i[0] for i in self.cur.description], 'data': self.cur.fetchall()}
        return pd.DataFrame(columns=data['header'], data=data['data'])

    def close_conn(self):
        self.cur.close()
        self.con.close()