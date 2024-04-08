import numpy as np

from compiler.lib.write_data import coe2bin
from compiler.utils.compare_utils import coe_comparison
from compiler.utils.ins_utils import instruction_parsing

# instruction_parsing("./ins/auto_ins1.dat")
# instruction_parsing("./ins_obb.dat")
# instruction_parsing("./ins/auto_ins143.dat")
# instruction_parsing("../yolo_obb/123/ins.dat")
# instruction_parsing("./ins/auto_ins142.dat")
# instruction_parsing("./ins/auto_ins8.dat")
# instruction_parsing("./ins/auto_ins9.dat")

# instruction_parsing("./ins.dat")

coe_comparison("./simulate_result/auto_simulate14.coe","./mid_result/auto_result14.coe")
# coe_comparison("./simulate_result/auto_simulate1.coe","../yolo_obb/fpga_1.coe")
# coe_comparison("./mid_result/auto_result1.coe","../yolo_obb/fpga_1.coe")
# coe_comparison("./simulate_result/auto_simulate3.coe", "../compiler/3.coe")


# coe2bin("input_add.coe","input_add.bin")
# coe2bin("weight_block2.coe","weight_block2.bin")
# coe2bin("weight_block3.coe","weight_block3.bin")


def bin2coe(binpath, coepath):
    target = open(coepath, "w")
    binfile = open(binpath, 'rb')
    ch = binfile.read(1)
    out = []
    while ch:
        data = ord(ch)
        # print(data)
        out.append(data)
        # target.write("%02X" % (data))
        if len(out) % 16 == 0:
            out.reverse()
            for i in out:
                target.write("%02X" % (i))
            target.write("\n")
            out = []

        ch = binfile.read(1)
    target.close()
    binfile.close()

# bin2coe("weight.bin","weight.coe")
