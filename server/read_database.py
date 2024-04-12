import pymysql
import pandas as pd

class SlurmDataBase:
    def __init__(self, host, user, password, database, cluster_prefix):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.cluster_prefix = cluster_prefix
    
    def read_table(self, query, query_params={}, df=True, verbose=False):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        cursor = conn.cursor()
    
        final_query = query.format(**query_params)
        if verbose:
            print(final_query)
        cursor.execute(final_query)
        rows = cursor.fetchall()
        result = rows if not df else pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
        
        cursor.close()
        conn.close()
        return result
    
    def read_running_jobs(self, columns=["*"], df=True, verbose=False):
        RUNNING_TASKS_QUERY = "select {columns} from {cluste_prefix}_job_table where time_start > 0 and time_end = 0"
        return self.read_table(RUNNING_TASKS_QUERY, {"columns": ', '.join(columns), "cluste_prefix": self.cluster_prefix}, df=df, verbose=verbose)

    def read_queued_jobs(self, columns=["*"], df=True, verbose=False):
        RUNNING_TASKS_QUERY = "select {columns} from {cluste_prefix}_job_table where time_submit > 0 and time_start = 0"
        return self.read_table(RUNNING_TASKS_QUERY, {"columns": ', '.join(columns), "cluste_prefix": self.cluster_prefix}, df=df, verbose=verbose)
    
    