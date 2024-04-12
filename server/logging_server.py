
import csv

from flask import Flask, request, jsonify
from .read_database import SlurmDataBase

app = Flask(__name__)
app.config["LOG_CSV_FILE"] = "default_log_file.csv"
app.config["DATABASE_PARAMS"] = {
    "host": "localhost",
    "user": "slurm",
    "password": "slurm",
    "database": "slurmdb_micro3",
    "cluster_prefix": "micro"
}

CURRENT_TASK_FACTORS = [
    # "job_id", 
    "cpus_per_task", "min_cpus", "max_cpus",
    # "min_nodes", "max_nodes", "pn_min_cpus", "ntasks_per_node",
    # "pn_min_memory", "time_limit"
]

database = SlurmDataBase()
def agregate_queue(

@app.route('/', methods=['POST'])
def handle_job_submit():
    data = request.get_json()
    current_task_factors = {k: v for k, v in data.items() if k in CURRENT_TASK_FACTORS}
    
    queue_tasks = 
    with open(app.config["LOG_CSV_FILE"], 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CURRENT_TASK_FACTORS)
        writer.writerow(current_task_factors)
      
    
    response = {
        'status': 'success',
        'message': 'The job description has been received and successfully written.'
    }
    return jsonify(response), 200
