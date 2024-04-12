import csv

from flask import Flask, request, jsonify
from .read_database import SlurmDataBase
from .utils import compute_aggregated_numeric_stats

CURRENT_TASK_FACTORS = [
    "job_id", 
    "cpus_per_task", "min_cpus", "max_cpus",
    "min_nodes", "max_nodes", "pn_min_cpus", "ntasks_per_node",
    "pn_min_memory", "time_limit"
]

class LoggingServer:
    def __init__(self, csv_file, database_params):
        self.slurm_db = SlurmDataBase(**database_params)
        self.csv_file = csv_file
        self.first_record = True
    
    def _get_queue_factors(self):
        queue_data = self.slurm_db.read_queued_jobs(columns=["cpus_req", "mem_req"], df=True)
        result = {k + "_queue": v for k, v in compute_aggregated_numeric_stats(queue_data).items()}
        result["count_queue"] = queue_data.shape[0]
        return result
    
    def _get_running_factors(self):
        running_data = self.slurm_db.read_running_jobs(columns=["cpus_req", "mem_req"], df=True)
        result = {k + "_running": v for k, v in compute_aggregated_numeric_stats(running_data).items()}
        result["count_running"] = running_data.shape[0]
        return result

    def start(self, port):
        self.app = Flask("LoggingServer")

        @self.app.route('/', methods=['POST'])
        def handle_job_submit():
            data = request.get_json()
            
            current_task_factors = {k: v for k, v in data.items() if k in CURRENT_TASK_FACTORS}
            queue_factors = self._get_queue_factors()
            running_factors = self._get_running_factors()
            
            full_factors = {**current_task_factors, **queue_factors, **running_factors}

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