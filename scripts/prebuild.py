import shutil


fp_out_name = "src/gbgpu/utils/constants.py"
fp_in_name = "src/gbgpu/cutils/include/Constants.h"

# develop few.utils.constants.py
with open(fp_out_name, "w") as fp_out:
    with open(fp_in_name, "r") as fp_in:
        lines = fp_in.readlines()
        for line in lines:
            if len(line.split()) == 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except ValueError as e:
                        continue


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
