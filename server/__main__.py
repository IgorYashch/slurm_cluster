import argparse
import yaml
from .simple_server import app as simple_server_app
from .logging_server2 import LoggingServer

parser = argparse.ArgumentParser(description='Launch the server with the specified mode.')
parser.add_argument('--port', type=int, help='Port on which the server is running', default=4567)
parser.add_argument('--slurmdb_config', type=str, help='Path to slurm database params', default="/root/server/slurmdb_params.yaml")

subparsers = parser.add_subparsers(dest='mode', help='Server launch modes')
subparsers.required = True

parser_simple = subparsers.add_parser('simple', help='Launch in simple mode')

parser_logging = subparsers.add_parser('logging', help='Launch in logging mode')
parser_logging.add_argument('--log_file', type=str, help='File for logging', required=True)

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == 'simple':
        app = simple_server_app

    elif args.mode == 'logging':
        with open(args.slurmdb_config, 'r') as file:
            slurm_db_params = yaml.safe_load(file)

        server = LoggingServer(args.log_file, database_params=slurm_db_params)

    server.start(port=args.port)
