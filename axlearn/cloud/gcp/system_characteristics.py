# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/xpk:
# Copyright 2023 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").

"""GCP system characteristics."""

from dataclasses import dataclass

AcceleratorType = {"TPU": 1, "GPU": 2, "CPU": 3}


@dataclass
class _SystemCharacteristics:
    topology: str
    vms_per_slice: int
    gke_accelerator: str
    gce_machine_type: str
    chips_per_vm: int
    accelerator_type: int
    device_type: str


USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS = {
    # GPU system characteristics
    # A100-40gb-$CHIPS
    "a100-40gb-1": _SystemCharacteristics(
        "N/A", 1, "nvidia-tesla-a100", "a2-highgpu-1g", 1, AcceleratorType["GPU"], "a100-40gb-1"
    ),
    "a100-40gb-2": _SystemCharacteristics(
        "N/A", 1, "nvidia-tesla-a100", "a2-highgpu-2g", 2, AcceleratorType["GPU"], "a100-40gb-2"
    ),
    "a100-40gb-4": _SystemCharacteristics(
        "N/A", 1, "nvidia-tesla-a100", "a2-highgpu-4g", 4, AcceleratorType["GPU"], "a100-40gb-4"
    ),
    "a100-40gb-8": _SystemCharacteristics(
        "N/A", 1, "nvidia-tesla-a100", "a2-highgpu-8g", 8, AcceleratorType["GPU"], "a100-40gb-8"
    ),
    # H100-80gb-$CHIPS
    "h100-80gb-8": _SystemCharacteristics(
        "N/A", 1, "nvidia-h100-80gb", "a3-highgpu-8g", 8, AcceleratorType["GPU"], "h100-80gb-8"
    ),
    # TPU system characteristics
    # v5p
    "v5p-8": _SystemCharacteristics(
        "2x2x1", 1, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8"
    ),
    "v5p-16": _SystemCharacteristics(
        "2x2x2", 2, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-16"
    ),
    "v5p-32": _SystemCharacteristics(
        "2x2x4", 4, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-32"
    ),
    "v5p-64": _SystemCharacteristics(
        "2x4x4", 8, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-64"
    ),
    "v5p-128": _SystemCharacteristics(
        "4x4x4", 16, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-128"
    ),
    "v5p-256": _SystemCharacteristics(
        "4x4x8", 32, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-256"
    ),
    "v5p-384": _SystemCharacteristics(
        "4x4x12", 48, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-384"
    ),
    "v5p-512": _SystemCharacteristics(
        "4x8x8", 64, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-512"
    ),
    "v5p-640": _SystemCharacteristics(
        "4x4x20", 80, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-640"
    ),
    "v5p-768": _SystemCharacteristics(
        "4x8x12", 96, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-768"
    ),
    "v5p-896": _SystemCharacteristics(
        "4x4x28", 112, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-896"
    ),
    "v5p-1024": _SystemCharacteristics(
        "8x8x8", 128, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1024"
    ),
    "v5p-1152": _SystemCharacteristics(
        "4x12x12", 144, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1152"
    ),
    "v5p-1280": _SystemCharacteristics(
        "4x8x20", 160, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1280"
    ),
    "v5p-1408": _SystemCharacteristics(
        "4x4x44", 176, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1408"
    ),
    "v5p-1536": _SystemCharacteristics(
        "8x8x12", 192, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1536"
    ),
    "v5p-1664": _SystemCharacteristics(
        "4x4x52", 208, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1664"
    ),
    "v5p-1792": _SystemCharacteristics(
        "4x8x28", 224, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1792"
    ),
    "v5p-1920": _SystemCharacteristics(
        "4x12x20", 240, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-1920"
    ),
    "v5p-2048": _SystemCharacteristics(
        "8x8x16", 256, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2048"
    ),
    "v5p-2176": _SystemCharacteristics(
        "4x4x68", 272, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2176"
    ),
    "v5p-2304": _SystemCharacteristics(
        "8x12x12", 288, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2304"
    ),
    "v5p-2432": _SystemCharacteristics(
        "4x4x76", 304, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2432"
    ),
    "v5p-2560": _SystemCharacteristics(
        "8x8x20", 320, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2560"
    ),
    "v5p-2688": _SystemCharacteristics(
        "4x12x28", 336, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2688"
    ),
    "v5p-2816": _SystemCharacteristics(
        "4x8x44", 352, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2816"
    ),
    "v5p-2944": _SystemCharacteristics(
        "4x4x92", 368, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-2944"
    ),
    "v5p-3072": _SystemCharacteristics(
        "8x12x16", 384, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3072"
    ),
    "v5p-3200": _SystemCharacteristics(
        "4x20x20", 400, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3200"
    ),
    "v5p-3328": _SystemCharacteristics(
        "4x8x52", 416, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3328"
    ),
    "v5p-3456": _SystemCharacteristics(
        "12x12x12", 432, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3456"
    ),
    "v5p-3584": _SystemCharacteristics(
        "8x8x28", 448, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3584"
    ),
    "v5p-3712": _SystemCharacteristics(
        "4x4x116", 464, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3712"
    ),
    "v5p-3840": _SystemCharacteristics(
        "8x12x20", 480, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3840"
    ),
    "v5p-3968": _SystemCharacteristics(
        "4x4x124", 496, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-3968"
    ),
    "v5p-4096": _SystemCharacteristics(
        "8x16x16", 512, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4096"
    ),
    "v5p-4224": _SystemCharacteristics(
        "4x12x44", 528, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4224"
    ),
    "v5p-4352": _SystemCharacteristics(
        "4x8x68", 544, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4352"
    ),
    "v5p-4480": _SystemCharacteristics(
        "4x20x28", 560, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4480"
    ),
    "v5p-4608": _SystemCharacteristics(
        "12x12x16", 576, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4608"
    ),
    "v5p-4736": _SystemCharacteristics(
        "4x4x148", 592, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4736"
    ),
    "v5p-4864": _SystemCharacteristics(
        "4x8x76", 608, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4864"
    ),
    "v5p-4992": _SystemCharacteristics(
        "4x12x52", 624, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-4992"
    ),
    "v5p-5120": _SystemCharacteristics(
        "8x16x20", 640, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5120"
    ),
    "v5p-5248": _SystemCharacteristics(
        "4x4x164", 656, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5248"
    ),
    "v5p-5376": _SystemCharacteristics(
        "8x12x28", 672, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5376"
    ),
    "v5p-5504": _SystemCharacteristics(
        "4x4x172", 688, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5504"
    ),
    "v5p-5632": _SystemCharacteristics(
        "8x8x44", 704, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5632"
    ),
    "v5p-5760": _SystemCharacteristics(
        "12x12x20", 720, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5760"
    ),
    "v5p-5888": _SystemCharacteristics(
        "4x8x92", 736, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-5888"
    ),
    "v5p-6016": _SystemCharacteristics(
        "4x4x188", 752, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6016"
    ),
    "v5p-6144": _SystemCharacteristics(
        "12x16x16", 768, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6144"
    ),
    "v5p-6272": _SystemCharacteristics(
        "4x28x28", 784, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6272"
    ),
    "v5p-6400": _SystemCharacteristics(
        "8x20x20", 800, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6400"
    ),
    "v5p-6528": _SystemCharacteristics(
        "4x12x68", 816, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6528"
    ),
    "v5p-6656": _SystemCharacteristics(
        "8x8x52", 832, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6656"
    ),
    "v5p-6784": _SystemCharacteristics(
        "4x4x212", 848, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6784"
    ),
    "v5p-6912": _SystemCharacteristics(
        "12x12x24", 864, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-6912"
    ),
    "v5p-7040": _SystemCharacteristics(
        "4x20x44", 880, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7040"
    ),
    "v5p-7168": _SystemCharacteristics(
        "8x16x28", 896, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7168"
    ),
    "v5p-7296": _SystemCharacteristics(
        "4x12x76", 912, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7296"
    ),
    "v5p-7424": _SystemCharacteristics(
        "4x8x116", 928, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7424"
    ),
    "v5p-7552": _SystemCharacteristics(
        "4x4x236", 944, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7552"
    ),
    "v5p-7680": _SystemCharacteristics(
        "12x16x20", 960, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7680"
    ),
    "v5p-7808": _SystemCharacteristics(
        "4x4x244", 976, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7808"
    ),
    "v5p-7936": _SystemCharacteristics(
        "4x8x124", 992, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-7936"
    ),
    "v5p-8064": _SystemCharacteristics(
        "12x12x28", 1008, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8064"
    ),
    "v5p-8192": _SystemCharacteristics(
        "16x16x16", 1024, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8192"
    ),
    "v5p-8320": _SystemCharacteristics(
        "4x20x52", 1040, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8320"
    ),
    "v5p-8448": _SystemCharacteristics(
        "8x12x44", 1056, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8448"
    ),
    "v5p-8704": _SystemCharacteristics(
        "8x8x68", 1088, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8704"
    ),
    "v5p-8832": _SystemCharacteristics(
        "4x12x92", 1104, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8832"
    ),
    "v5p-8960": _SystemCharacteristics(
        "8x20x28", 1120, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-8960"
    ),
    "v5p-9216": _SystemCharacteristics(
        "12x16x24", 1152, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9216"
    ),
    "v5p-9472": _SystemCharacteristics(
        "4x8x148", 1184, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9472"
    ),
    "v5p-9600": _SystemCharacteristics(
        "12x20x20", 1200, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9600"
    ),
    "v5p-9728": _SystemCharacteristics(
        "8x8x76", 1216, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9728"
    ),
    "v5p-9856": _SystemCharacteristics(
        "4x28x44", 1232, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9856"
    ),
    "v5p-9984": _SystemCharacteristics(
        "8x12x52", 1248, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-9984"
    ),
    "v5p-10240": _SystemCharacteristics(
        "16x16x20", 1280, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-10240"
    ),
    "v5p-10368": _SystemCharacteristics(
        "12x12x36", 1296, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-10368"
    ),
    "v5p-10496": _SystemCharacteristics(
        "4x8x164", 1312, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-10496"
    ),
    "v5p-10752": _SystemCharacteristics(
        "12x16x28", 1344, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-10752"
    ),
    "v5p-10880": _SystemCharacteristics(
        "4x20x68", 1360, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-10880"
    ),
    "v5p-11008": _SystemCharacteristics(
        "4x8x172", 1376, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11008"
    ),
    "v5p-11136": _SystemCharacteristics(
        "4x12x116", 1392, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11136"
    ),
    "v5p-11264": _SystemCharacteristics(
        "8x16x44", 1408, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11264"
    ),
    "v5p-11520": _SystemCharacteristics(
        "12x20x24", 1440, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11520"
    ),
    "v5p-11648": _SystemCharacteristics(
        "4x28x52", 1456, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11648"
    ),
    "v5p-11776": _SystemCharacteristics(
        "8x8x92", 1472, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11776"
    ),
    "v5p-11904": _SystemCharacteristics(
        "4x12x124", 1488, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-11904"
    ),
    "v5p-12032": _SystemCharacteristics(
        "4x8x188", 1504, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-12032"
    ),
    "v5p-12160": _SystemCharacteristics(
        "4x20x76", 1520, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-12160"
    ),
    "v5p-12288": _SystemCharacteristics(
        "16x16x24", 1536, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-12288"
    ),
    "v5p-13824": _SystemCharacteristics(
        "12x24x24", 1728, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-13824"
    ),
    "v5p-17920": _SystemCharacteristics(
        "16x20x28", 2240, "tpu-v5p-slice", "ct5p-hightpu-4t", 4, AcceleratorType["TPU"], "v5p-17920"
    ),
    # v5litepod
    "v5litepod-16": _SystemCharacteristics(
        "4x4",
        4,
        "tpu-v5-lite-podslice",
        "ct5lp-hightpu-4t",
        4,
        AcceleratorType["TPU"],
        "v5litepod-16",
    ),
    "v5litepod-32": _SystemCharacteristics(
        "4x8",
        8,
        "tpu-v5-lite-podslice",
        "ct5lp-hightpu-4t",
        4,
        AcceleratorType["TPU"],
        "v5litepod-32",
    ),
    "v5litepod-64": _SystemCharacteristics(
        "8x8",
        16,
        "tpu-v5-lite-podslice",
        "ct5lp-hightpu-4t",
        4,
        AcceleratorType["TPU"],
        "v5litepod-64",
    ),
    "v5litepod-128": _SystemCharacteristics(
        "8x16",
        32,
        "tpu-v5-lite-podslice",
        "ct5lp-hightpu-4t",
        4,
        AcceleratorType["TPU"],
        "v5litepod-128",
    ),
    "v5litepod-256": _SystemCharacteristics(
        "16x16",
        64,
        "tpu-v5-lite-podslice",
        "ct5lp-hightpu-4t",
        4,
        AcceleratorType["TPU"],
        "v5litepod-256",
    ),
    # v6e
    "v6e-4": _SystemCharacteristics(
        "2x2", 1, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-4"
    ),
    "v6e-8": _SystemCharacteristics(
        "2x4", 2, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-8"
    ),
    "v6e-16": _SystemCharacteristics(
        "4x4", 4, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-16"
    ),
    "v6e-32": _SystemCharacteristics(
        "4x8", 8, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-32"
    ),
    "v6e-64": _SystemCharacteristics(
        "8x8", 16, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-64"
    ),
    "v6e-128": _SystemCharacteristics(
        "8x16", 32, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-128"
    ),
    "v6e-256": _SystemCharacteristics(
        "16x16", 64, "tpu-v6e-slice", "ct6e-standard-4t", 4, AcceleratorType["TPU"], "v6e-256"
    ),
    # v4
    "v4-8": _SystemCharacteristics(
        "2x2x1", 1, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-8"
    ),
    "v4-16": _SystemCharacteristics(
        "2x2x2", 2, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-16"
    ),
    "v4-32": _SystemCharacteristics(
        "2x2x4", 4, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-32"
    ),
    "v4-64": _SystemCharacteristics(
        "2x4x4", 8, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-64"
    ),
    "v4-128": _SystemCharacteristics(
        "4x4x4", 16, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-128"
    ),
    "v4-256": _SystemCharacteristics(
        "4x4x8", 32, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-256"
    ),
    "v4-512": _SystemCharacteristics(
        "4x8x8", 64, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-512"
    ),
    "v4-1024": _SystemCharacteristics(
        "8x8x8", 128, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-1024"
    ),
    "v4-1536": _SystemCharacteristics(
        "8x8x12", 192, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-1536"
    ),
    "v4-2048": _SystemCharacteristics(
        "8x8x16", 256, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-2048"
    ),
    "v4-4096": _SystemCharacteristics(
        "8x16x16", 512, "tpu-v4-podslice", "ct4p-hightpu-4t", 4, AcceleratorType["TPU"], "v4-4096"
    ),
    # CPU system characteristics
    # n2-standard-32-$VMs
    "n2-standard-32-1": _SystemCharacteristics(
        "N/A", 1, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-1"
    ),
    "n2-standard-32-2": _SystemCharacteristics(
        "N/A", 2, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-2"
    ),
    "n2-standard-32-4": _SystemCharacteristics(
        "N/A", 4, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-4"
    ),
    "n2-standard-32-8": _SystemCharacteristics(
        "N/A", 8, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-8"
    ),
    "n2-standard-32-16": _SystemCharacteristics(
        "N/A", 16, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-16"
    ),
    "n2-standard-32-32": _SystemCharacteristics(
        "N/A", 32, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-32"
    ),
    "n2-standard-32-64": _SystemCharacteristics(
        "N/A", 64, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-64"
    ),
    "n2-standard-32-128": _SystemCharacteristics(
        "N/A", 128, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-128"
    ),
    "n2-standard-32-256": _SystemCharacteristics(
        "N/A", 256, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-256"
    ),
    "n2-standard-32-512": _SystemCharacteristics(
        "N/A", 512, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-512"
    ),
    "n2-standard-32-1024": _SystemCharacteristics(
        "N/A", 1024, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-1024"
    ),
    "n2-standard-32-2048": _SystemCharacteristics(
        "N/A", 2048, "N/A", "n2-standard-32", 1, AcceleratorType["CPU"], "n2-standard-32-2048"
    ),
}

# Reference doc https://cloud.google.com/tpu/docs/tpus-in-gke.
GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS = {
    "ct5p-hightpu-4t": 448,
    "ct4p-hightpu-4t": 407,
    "ct5lp-hightpu-4t": 192,
}
