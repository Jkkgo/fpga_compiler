from compiler.lib.write_data import coe2bin
from compiler.utils.ins_utils import instruction_parsing



# instruction_parsing("./ins/auto_ins1.dat")
# instruction_parsing("./ins_obb.dat")
# instruction_parsing("./ins/auto_ins5.dat")
# instruction_parsing("./ins/auto_ins6.dat")
# instruction_parsing("./ins/auto_ins7.dat")
# instruction_parsing("./ins/auto_ins8.dat")
# instruction_parsing("./ins/auto_ins9.dat")

# instruction_parsing("./ins.dat")

# coe_comparison("./weight/weight1.coe","./weight/weight1_real.coe")
coe2bin("weight_block1.coe","weight_block1.bin")
coe2bin("weight_block2.coe","weight_block2.bin")
coe2bin("weight_block3.coe","weight_block3.bin")


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



