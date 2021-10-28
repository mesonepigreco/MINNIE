import pytest
import minnie, minnie.Atoms as Atoms
import minnie.SymmetricFunctions as SF


def test_symmetricget():
    sf = SF.SymmetricFunctions()
    sf.load_from_cfg("../../ReadEnsemble/input_other_ensemble.cfg")


    ng2 = sf.get_number_of_g2()
    ng4 = sf.get_number_of_g4()

    print("We find {} g2 and {} g4 functions.".format(ng2, ng4))

    t, rc = sf.get_cutoff_function()

    print("We employed a cutoff type: {} with Rc = {} A".format(t, rc))

    for i in range(ng2):
        params = sf.get_parameters(i, "g2")
        print ("Parameters of function {}:".format(i))
        print(params)
        
    for i in range(ng4):
        params = sf.get_parameters(i, "g4")
        print ("Parameters of function {}:".format(i))
        print(params)

    # Change the parameters of the first g4
    sf.set_parameters(0, {"g4_zeta" : 50})
    sf.set_cutoff_radius(10)
    sf.set_cutoff_function_type(1)

    sf.print_info()
    
    sf.add_g2_function(0.4, 1)    
    print("HERE")
    sf.add_g4_function(0.8, 1.2, 2)
    print("HERE")
    sf.save_to_cfg("new_symf.cfg")

    

        

if __name__ == "__main__":
    test_symmetricget()
