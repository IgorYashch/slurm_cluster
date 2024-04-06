#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <inttypes.h>
#include <sys/types.h>
#include <unistd.h>
#include "slurm/slurm.h"
#include "slurm/slurm_errno.h"
#include "src/slurmctld/slurmctld.h"

#include "utils.h"

const char plugin_name[] = "Plugin for predict waittime";
const char plugin_type[] = "job_submit/predict";
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

// Callback function to receive the response from the server
size_t write_callback(void *contents, size_t size, size_t nmemb, char **response)
{
    size_t real_size = size * nmemb;
    char *ptr = realloc(*response, strlen(*response) + real_size + 1);
    if (!ptr) {
        return 0; // Failed to realloc memory, return 0 to signal error
    }

    *response = ptr;
    memcpy(*response + strlen(*response), contents, real_size);
    (*response)[strlen(*response) + real_size] = '\0';

    return real_size;
}

extern int job_submit(job_desc_msg_t *job_desc, uint32_t submit_uid, char **err_msg)
{
    const char *argument_for_plugin = "predict-time";

    // Если указан комментарий для прогнозирования, то выдаем обратно ошибку и печатаем предсказаение в сообщение об ошибке
    if (job_desc->comment && strcmp(job_desc->comment, argument_for_plugin) == 0)
    {
        char *json_message = description_to_json(job_desc);

        CURL *curl = curl_easy_init();
        if (curl)
        {
            char *response = strdup(""); // Initialize empty string for response
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:4567");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_message);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK)
            {
                // Add the server response to err_msg
                *err_msg = strdup(response);
            }
            else
            {
                *err_msg = strdup(curl_easy_strerror(res));
            }

            free(response);
            curl_easy_cleanup(curl);
        }
        else
        {
            *err_msg = strdup("Failed to initialize CURL.");
        }

        free(json_message);
        return SLURM_ERROR;
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
    info("%s plugin loaded", plugin_name);

    CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
    if (res != CURLE_OK)
    {
        error("Failed to initialize libcurl: %s", curl_easy_strerror(res));
        return SLURM_ERROR;
    }

    return SLURM_SUCCESS;
}

extern int fini(void)
{
    info("%s plugin unloaded", plugin_name);
    curl_global_cleanup();
    return SLURM_SUCCESS;
}
