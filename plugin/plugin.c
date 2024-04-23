#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <slurm/slurm.h>
#include <slurm/slurm_errno.h>
#include "src/slurmctld/slurmctld.h"
#include "utils.h"

const char plugin_name[] = "Plugin for predict waittime";
const char plugin_type[] = "job_submit/predict";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

// Callback function to receive the response from the server
size_t write_callback(void *contents, size_t size, size_t nmemb, char **response)
{
    size_t total_size = size * nmemb;
    *response = realloc(*response, total_size + 1);
    if (*response)
    {
        memcpy(*response, contents, total_size);
        (*response)[total_size] = '\0';
    }
    return total_size;
}

extern int job_submit(job_desc_msg_t *job_desc, uint32_t submit_uid, char **err_msg) {
    const char *predict_time_arg = "predict-time";
    const char *logging_arg = "logging";

    if (job_desc->comment && (strcmp(job_desc->comment, predict_time_arg) == 0 || strcmp(job_desc->comment, logging_arg) == 0)) {
        char *json_message = description_to_json(job_desc);

        CURL *curl = curl_easy_init();
        if (curl) {
            char *response = NULL;
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:4567");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_message);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK) {
                if (strcmp(job_desc->comment, logging_arg) == 0) {
                    info("Logging mode: Sent message: %s\n", json_message);
                } else if (strcmp(job_desc->comment, predict_time_arg) == 0) {
                    // Предполагаем, что ответ сервера должен быть записан в сообщение об ошибке
                    if (response) {
                        *err_msg = strdup(response);
                        curl_slist_free_all(headers);
                        curl_free(response);
                        curl_easy_cleanup(curl);
                        free(json_message);
                        return SLURM_ERROR;
                    }
                }
            } else {
                info("Failed to send message.\n");
            }

            curl_slist_free_all(headers);
            curl_free(response);
            curl_easy_cleanup(curl);
        }

        free(json_message);
    }

    return SLURM_SUCCESS;
}


extern int job_modify(job_desc_msg_t *job_desc, job_record_t *job_ptr, uint32_t submit_uid)
{
    info("Job %u modified by user %u", job_desc->job_id, submit_uid);
    return SLURM_SUCCESS;
}

extern int init(void)
{
    info("job_submit plugin loaded");

    // Initialize curl globally
    CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (res != CURLE_OK)
    {
        fprintf(stderr, "Failed to initialize libcurl: %s\n", curl_easy_strerror(res));
        return SLURM_ERROR;
    }

    return SLURM_SUCCESS;
}

extern int fini(void)
{
    info("job_submit plugin unloaded");

    curl_global_cleanup();

    return SLURM_SUCCESS;
}
