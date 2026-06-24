idf_component_register(
    SRCS
        "camera_web_main.c"
        "avi_mjpeg_writer.c"
        "legacy_recognition_stubs.cpp"
        "stub_embedded_assets.c"
        "tiny_cls_espdl_bridge.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        esp-dl
        esp_event
        esp_http_server
        esp_netif
        esp_system
        esp_timer
        esp_wifi
        esp_video
        example_video_common
        fatfs
        wear_levelling
        sdmmc
        esp_driver_sdmmc
        esp_driver_sdspi
        esp_driver_spi
        esp_driver_gpio
        vfs
        nvs_flash
)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
else()
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Tiny CNN 海面目标 6 类分类模型。
# 导出路线：TinyCNN PT -> ONNX -> ESP-PPQ/ESP-DL INT8 -> .espdl。
# 输入为 NHWC 1x96x96x3，类别为 unknown/plastic_bottle/foam/buoy/net/ship_part。
set(TINY_CLS_MODEL_FILE "${CMAKE_CURRENT_LIST_DIR}/../models/tiny_cls_96_6cls_s8_p4.espdl")
target_add_aligned_binary_data(${COMPONENT_LIB} "${TINY_CLS_MODEL_FILE}" BINARY)
