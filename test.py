import sqlite3


dataset_path = 'database/database.db'

con = sqlite3.connect(dataset_path)
cur = con.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS EMPLOYESS (
    EMP_ID      INTEGER     NOT NULL    PRIMARY KEY AUTOINCREMENT   UNIQUE,
    EMP_NAME    TEXT        NOT NULL,
    EMP_DEPT_ID INT         NOT NULL
    ); '''
)
cur.execute('''CREATE TABLE IF NOT EXISTS EMP_EMBS (
    FACE        TEXT        NOT NULL,
    EMB         TEXT        NOT NULL,
    MASKED_EMB  TEXT        NOT NULL,
    EMP_ID      INTERGER    NOT NULL,
    FOREIGN KEY (EMP_ID)    REFERENCES EMPLOYESS (EMP_ID)   ON DELETE CASCADE
    ); '''
)
cur.execute('''CREATE TABLE IF NOT EXISTS EMP_ATT (
    STATUS      TEXT        NOT NULL,
    CREATED     TEXT        NOT NULL,
    EMP_ID      INTERGER    NOT NULL,
    FOREIGN KEY (EMP_ID)    REFERENCES EMPLOYESS (EMP_ID)   ON DELETE CASCADE
    ); '''
)


# cur.execute('''INSERT INTO EMPLOYESS (EMP_NAME, EMP_DEPT_ID) VALUES (?, ?)''', ('emp_name', 1))
# con.commit()
# emp_id = cur.lastrowid
# cur.execute('''INSERT INTO EMP_EMBS (EMP_ID, FACE, EMB, MASKED_EMB) VALUES (?, ?, ?, ?)''', (3, 'face_json', 'emb_json', 'masked_emb_json'))
# con.commit()
cur.execute('''DELETE FROM EMPLOYESS WHERE EMP_ID=?''', ('3'))
con.commit()