#include "coco_espdl_bridge.h"
#include "yolo11_espdl_bridge.h"
#include "yolo26_espdl_bridge.h"

#include <string.h>

extern "C" {

bool yolo11_espdl_available(void)
{
    return false;
}

uint32_t yolo11_espdl_model_bytes(void)
{
    return 0;
}

esp_err_t yolo11_espdl_detect_jpeg(const uint8_t *jpg_data,
                                   size_t jpg_len,
                                   yolo11_espdl_result_t *out)
{
    (void)jpg_data;
    (void)jpg_len;
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    return ESP_ERR_NOT_SUPPORTED;
}

bool yolo26_espdl_available(void)
{
    return false;
}

uint32_t yolo26_espdl_model_bytes(void)
{
    return 0;
}

esp_err_t yolo26_espdl_detect_jpeg(const uint8_t *jpg_data,
                                   size_t jpg_len,
                                   yolo26_espdl_result_t *out)
{
    (void)jpg_data;
    (void)jpg_len;
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    return ESP_ERR_NOT_SUPPORTED;
}

bool coco_espdl_available(void)
{
    return false;
}

uint32_t coco_espdl_model_bytes(void)
{
    return 0;
}

esp_err_t coco_espdl_detect_jpeg(const uint8_t *jpg_data,
                                 size_t jpg_len,
                                 coco_espdl_result_t *out)
{
    (void)jpg_data;
    (void)jpg_len;
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t coco_espdl_detect_and_annotate_jpeg(const uint8_t *jpg_data,
                                              size_t jpg_len,
                                              uint32_t min_score,
                                              uint8_t jpeg_quality,
                                              coco_espdl_result_t *out,
                                              uint8_t **annotated_jpeg,
                                              size_t *annotated_len)
{
    (void)jpg_data;
    (void)jpg_len;
    (void)min_score;
    (void)jpeg_quality;
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (annotated_jpeg) {
        *annotated_jpeg = nullptr;
    }
    if (annotated_len) {
        *annotated_len = 0;
    }
    return ESP_ERR_NOT_SUPPORTED;
}

void coco_espdl_free_jpeg(uint8_t *jpeg)
{
    (void)jpeg;
}

}
