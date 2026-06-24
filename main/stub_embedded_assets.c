#include <stdint.h>

#define STUB_ASSET(name) \
    const uint8_t name##_start[] asm("_binary_" #name "_start") = {0}; \
    const uint8_t name##_end[] asm("_binary_" #name "_end") = {0}

STUB_ASSET(coke_01_jpg);
STUB_ASSET(sprite_01_jpg);
STUB_ASSET(demo_01_jpg);
STUB_ASSET(demo_02_jpg);
STUB_ASSET(demo_03_jpg);
STUB_ASSET(demo_04_jpg);
STUB_ASSET(frame_00001_jpg);
STUB_ASSET(frame_00002_jpg);
STUB_ASSET(frame_00003_jpg);
STUB_ASSET(frame_00004_jpg);
STUB_ASSET(frame_00005_jpg);
STUB_ASSET(frame_00006_jpg);
STUB_ASSET(frame_00007_jpg);
STUB_ASSET(frame_00008_jpg);
STUB_ASSET(frame_00009_jpg);
STUB_ASSET(frame_00010_jpg);
STUB_ASSET(frame_00011_jpg);
STUB_ASSET(frame_00012_jpg);
STUB_ASSET(frame_00013_jpg);
STUB_ASSET(frame_00014_jpg);
STUB_ASSET(frame_00015_jpg);
STUB_ASSET(frame_00016_jpg);
