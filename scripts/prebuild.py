import shutil

cu_files = ["gbgpu_utils"]
pyx_files = ["GBGPU"]
for fp in cu_files:
    shutil.copy("src/gbgpu/cutils/src/" + fp + ".cu", "src/gbgpu/cutils/src/" + fp + ".cpp")

for fp in pyx_files:
    shutil.copy(
        "src/gbgpu/cutils/src/" + fp + ".pyx", "src/gbgpu/cutils/src/" + fp + "_cpu.pyx"
    )


# for fp in cu_files:
#     os.remove("src/" + fp + ".cpp")

# for fp in pyx_files:
#     os.remove("src/" + fp + "_cpu.pyx")
