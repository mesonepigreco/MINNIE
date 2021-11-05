"""
Extract the info on a single atomic neural network to test energies and forces on output.
"""

import sys, os
import numpy
import numpy as np
import matplotlib.pyplot as plt

def get_all_data(fname, ann_id, coord_id = 1):

    current_id = -1
    current_t = 0

    all_data = []
    data = {}
    toscalarg = []
    toscalarf = []
    symf = []
    first_layer = []
    first_layer_grad = []
    with open(fname, "r") as fp:
        read_grad = False
        read_dG_dX = False
        for line in fp.readlines():
            line = line.strip()
            if not line.startswith("#"):
                data["scalar_product"] = np.dot(toscalarg, toscalarf)
                all_data.append(data)
                current_t += 1
                current_id = -1
                data = {}
                toscalarg = []
                toscalarf = []


            if line.startswith("# SYM F [0]") or line.startswith("# Force on"):
                current_id += 1
                if current_id == ann_id:
                    symf = []
                    first_layer = []
                    first_layer_grad = []
                

            if line.startswith("# SYM F"):
                symf.append(float(line.split()[-1]))

            if line.startswith("# First layer ["):
                first_layer.append(float(line.split()[-1]))

            if line.startswith("# First layer grad"):
                first_layer_grad.append(float(line.split()[-1]))
                toscalarf.append(float(line.split()[-1]))
                read_grad = True
            else:
                if read_grad:
                    if current_id - 1 == ann_id:
                        data["gradient"] = np.array(first_layer_grad)
                read_grad = False

            if line.startswith("# Output"):
                if current_id == ann_id:
                    data["symmetric_functions"] = np.array(symf)
                    data["first_layer"] = np.array(first_layer)
                    data["output"] = float(line.split()[5]) * float(line.split()[-1])


            if line.startswith("# dG / dX"):
                new_id = int(line.split()[7])
                coord = int(line.split()[15])
                if coord == coord_id:
                    toscalarg.append(float(line.split()[-1]))
                    
                if new_id == ann_id and coord == coord_id:
                    if not read_dG_dX:
                        Glist = []

                    read_dG_dX = True

                    Glist.append( float(line.split()[-1]))
            else:
                if read_dG_dX:
                    data["dG_dX"] = np.array(Glist)
                read_dG_dX = False
                
            if line.startswith("# Force on atom [0] coord [%d]" % coord_id):
                data["force_projected"] = float(line.split()[-1])

    return all_data
                
                
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error, required 1 argument with the file to be analyzed.")
        exit()
        
    fname = sys.argv[1]
    if len(sys.argv) == 3:
        idd = int(sys.argv[2])
    else:
        idd = 0
    all_data = get_all_data(fname, idd)
    
    total_force = []
    scalar_product = []

    dE_anals = []
    dE_nums = []

    dG_anals = []
    dG_nums = []
    for t, data in enumerate(all_data):
        if t==0:
            continue
        old_data = all_data[t-1]

        ds = data["first_layer"] - old_data["first_layer"]
        f = .5 * (old_data["gradient"] + data["gradient"])
        
        total_force.append(data["force_projected"])
        scalar_product.append(data["scalar_product"])

        dE_anal = f.dot(ds)
        dE_num = data["output"] - old_data["output"]

        dE_anals.append(dE_anal)
        dE_nums.append(dE_num)

        dsym_num = data["symmetric_functions"] - old_data["symmetric_functions"]
        dsym_anal = .5 * (data["dG_dX"] + old_data["dG_dX"])

        dG_anals.append(dsym_anal)
        dG_nums.append(dsym_num / 1e-2)
        

    dE_nums = np.array(dE_nums)
    dE_anals = np.array(dE_anals)
    dG_anals = np.array(dG_anals)
    dG_nums = np.array(dG_nums)
    
    plt.figure()
    plt.plot(total_force, label = "F computed")
    plt.plot(scalar_product, label = "Scalar product")
    plt.legend()
    plt.title("Force comparison")
    plt.tight_layout()
    
    print("NF:", len(total_force))
    
    plt.figure()

    plt.plot(dE_anals, label = "dE analytic")
    plt.plot(-dE_nums, label = "dE numeric")
    plt.legend()
    plt.tight_layout()

    nsym = np.shape(dG_anals)[-1]
    nsym = 4 

    for k in range(nsym):
        plt.figure()
        plt.title("SYM ID {}".format(k))
        plt.plot(dG_anals[:, k], label = "dG analytic")
        plt.plot(dG_nums[:, k], label = "dG numeric")
        plt.legend()
        plt.tight_layout()

    plt.show()
