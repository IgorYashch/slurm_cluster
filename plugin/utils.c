#include "utils.h"
#include <stdio.h>
#include <stdlib.h>


char* description_to_json(struct job_descriptor *job) {
    char *json = malloc(2048 * sizeof(char));

    sprintf(json,
        "{\n"
        "  \"cpus_per_task\": %hu,\n"
        "  \"min_cpus\": %u,\n"
        "  \"max_cpus\": %u,\n"
        "  \"min_nodes\": %u,\n"
        "  \"max_nodes\": %u,\n"
        "  \"boards_per_node\": %hu,\n"
        "  \"sockets_per_board\": %hu,\n"
        "  \"sockets_per_node\": %hu,\n"
        "  \"cores_per_socket\": %hu,\n"
        "  \"threads_per_core\": %hu,\n"
        "  \"ntasks_per_node\": %hu,\n"
        "  \"ntasks_per_socket\": %hu,\n"
        "  \"ntasks_per_core\": %hu,\n"
        "  \"ntasks_per_board\": %hu,\n"
        "  \"pn_min_cpus\": %hu,\n"
        "  \"pn_min_memory\": %" PRIu64 ",\n"
        "  \"pn_min_tmp_disk\": %u,\n"
        "  \"req_switch\": %u,\n"
        "  \"wait4switch\": %u\n"
        "}",
        job->cpus_per_task,
        job->min_cpus,
        job->max_cpus,
        job->min_nodes,
        job->max_nodes,
        job->boards_per_node,
        job->sockets_per_board,
        job->sockets_per_node,
        job->cores_per_socket,
        job->threads_per_core,
        job->ntasks_per_node,
        job->ntasks_per_socket,
        job->ntasks_per_core,
        job->ntasks_per_board,
        job->pn_min_cpus,
        job->pn_min_memory,
        job->pn_min_tmp_disk,
        job->req_switch,
        job->wait4switch
    );

    return json;
}
