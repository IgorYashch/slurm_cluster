import csv

from flask import Flask, request, jsonify
from .read_database import SlurmDataBase
from .utils import compute_aggregated_numeric_stats
from datetime import datetime, timedelta
import numpy as np

CURRENT_TASK_FACTORS = [
    # "job_id", 
    "job_name",
    "cpus_per_task",
    "time_limit",
    "user_id"
    # "min_cpus", "max_cpus", "pn_min_memory"
    # "min_nodes", "max_nodes", "pn_min_cpus", "ntasks_per_node",
]


class PredictionServer:
    def __init__(self, model, csv_file, database_params):
        self.slurm_db = SlurmDataBase(**database_params)
        self.csv_file = csv_file
        self.first_record = True
    
    @staticmethod
    def _get_current_time():
        return datetime.now().timestamp()
    
    def _get_queue_factors(self, user_id):
        queue_data = self.slurm_db.read_queued_jobs(columns=["cpus_req", "mem_req", "timelimit", "time_submit", "id_user"], df=True)
        queue_data["now_time_waiting"] = self._get_current_time() - queue_data["time_submit"]
        queue_data_user = queue_data[queue_data["id_user"] == user_id]
        
        result = dict()
        result["count_queue"] = queue_data.shape[0]
        result["count_queue_user"] = queue_data_user.shape[0]
        
        result["mean_req_cpus"] = queue_data["cpus_req"].mean()
        result["sum_req_cpus"] = queue_data["cpus_req"].sum()
        result["min_req_cpus"] = queue_data["cpus_req"].min()
        result["max_req_cpus"] = queue_data["cpus_req"].max()
        result["mean_req_cpus_user"] = queue_data_user["cpus_req"].mean()
        result["sum_req_cpus_user"] = queue_data_user["cpus_req"].sum()
        result["min_req_cpus_user"] = queue_data_user["cpus_req"].min()
        result["max_req_cpus_user"] = queue_data_user["cpus_req"].max()
        
        result["mean_timelimit"] = queue_data["timelimit"].mean()
        result["sum_timelimit"] = queue_data["timelimit"].sum()
        result["min_timelimit"] = queue_data["timelimit"].min()
        result["max_timelimit"] = queue_data["timelimit"].max()
        result["mean_timelimit_user"] = queue_data_user["timelimit"].mean()
        result["sum_timelimit_user"] = queue_data_user["timelimit"].sum()
        result["min_timelimit_user"] = queue_data_user["timelimit"].min()
        result["max_timelimit_user"] = queue_data_user["timelimit"].max()
        
        result["mean_now_time_waiting"] = queue_data["now_time_waiting"].mean()
        result["sum_now_time_waiting"] = queue_data["now_time_waiting"].sum()
        result["min_now_time_waiting"] = queue_data["now_time_waiting"].min()
        result["max_now_time_waiting"] = queue_data["now_time_waiting"].max()
        result["mean_now_time_waiting_user"] = queue_data_user["now_time_waiting"].mean()
        result["sum_now_time_waiting_user"] = queue_data_user["now_time_waiting"].sum()
        result["min_now_time_waiting_user"] = queue_data_user["now_time_waiting"].min()
        result["max_now_time_waiting_user"] = queue_data_user["now_time_waiting"].max()
        
        for r in result:
            if np.isnan(result[r]):
                result[r] = 0
        return result
    
    def _get_running_factors(self, user_id):
        running_data = self.slurm_db.read_running_jobs(columns=["cpus_req", "mem_req", "time_start", "timelimit", "id_user"], df=True)
        running_data["now_time_running"] = self._get_current_time() - running_data["time_start"]
        running_data_user = running_data[running_data["id_user"] == user_id]
        
        result = dict()
        result["count_running"] = running_data.shape[0]
        result["count_running_user"] = running_data_user.shape[0]
        
        result["mean_req_cpus"] = running_data["cpus_req"].mean()
        result["sum_req_cpus"] = running_data["cpus_req"].sum()
        result["min_req_cpus"] = running_data["cpus_req"].min()
        result["max_req_cpus"] = running_data["cpus_req"].max()
        result["mean_req_cpus_user"] = running_data_user["cpus_req"].mean()
        result["sum_req_cpus_user"] = running_data_user["cpus_req"].sum()
        result["min_req_cpus_user"] = running_data_user["cpus_req"].min()
        result["max_req_cpus_user"] = running_data_user["cpus_req"].max()
        
        result["mean_timelimit"] = running_data["timelimit"].mean()
        result["sum_timelimit"] = running_data["timelimit"].sum()
        result["min_timelimit"] = running_data["timelimit"].min()
        result["max_timelimit"] = running_data["timelimit"].max()
        result["mean_timelimit_user"] = running_data_user["timelimit"].mean()
        result["sum_timelimit_user"] = running_data_user["timelimit"].sum()
        result["min_timelimit_user"] = running_data_user["timelimit"].min()
        result["max_timelimit_user"] = running_data_user["timelimit"].max()
        
        result["mean_now_time_running"] = running_data["now_time_running"].mean()
        result["sum_now_time_running"] = running_data["now_time_running"].sum()
        result["min_now_time_running"] = running_data["now_time_running"].min()
        result["max_now_time_running"] = running_data["now_time_running"].max()
        result["mean_now_time_running_user"] = running_data_user["now_time_running"].mean()
        result["sum_now_time_running_user"] = running_data_user["now_time_running"].sum()
        result["min_now_time_running_user"] = running_data_user["now_time_running"].min()
        result["max_now_time_running_user"] = running_data_user["now_time_running"].max()
        
        for r in result:
            if np.isnan(result[r]):
                result[r] = 0
        return result

    def start(self, port):
        self.app = Flask("LoggingServer")

        @self.app.route('/', methods=['POST'])
        def handle_job_submit():
            data = request.get_json()
            
            current_task_factors = {k: v for k, v in data.items() if k in CURRENT_TASK_FACTORS}
            # print(current_task_factors)
            queue_factors = self._get_queue_factors(data["user_id"])
            running_factors = self._get_running_factors(data["user_id"])
            
            full_factors = {**current_task_factors, **queue_factors, **running_factors}

            print(full_factors)
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(full_factors.keys()))
                
                if self.first_record:
                    writer.writeheader()
                    self.first_record = False
                
                writer.writerow(full_factors)

            response = {
                'status': 'success',
                'message': 'The job description has been received and successfully written.'
            }
            return jsonify(response), 200

        self.app.run(port=port)