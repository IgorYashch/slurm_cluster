#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <slurm/slurm.h>
#include <slurm/slurm_errno.h>
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

extern int job_submit(struct job_descriptor *job_desc, uint32_t submit_uid, char **err_msg)
{
    const char *argumet_for_plugin = "predict-time";

    // Если указан комментарий для прогнозирования, то выдаем обратно ошибку и печатаем предсказаение в сообщение об ошибке
    if (strcmp(job_desc->comment, argumet_for_plugin) == 0)
    {
        char *json_message = description_to_json(job_desc);

        CURL *curl = curl_easy_init();
        if (curl)
        {
            char *response = NULL;
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:4567");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_message);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK)
            {
                // Add the server response to err_msg
                *err_msg = realloc(*err_msg, strlen(response) + 1);
                strcpy(*err_msg, response);
            }
            else
            {
                fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            }

            curl_free(response);
            curl_easy_cleanup(curl);
        }

        free(json_message);
        return SLURM_ERROR;
    }

    return SLURM_SUCCESS;
}

extern int job_modify(struct job_descriptor *job_desc, struct job_record *job_ptr, uint32_t submit_uid)
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
