
all: NN.exe test_sd_deriv.exe test_ensemble.exe test_read_atoms.exe test_nn_prediction.exe

tests: test_nn_prediction.exe test_ensemble.exe test_sd_deriv.exe test_read_atoms.exe

NN.exe: main.cpp network.o utils.o test_read_atoms.exe test_ensemble.exe
	$(CPP) $(CFLAGS)  network.o utils.o  main.cpp  -o NN.exe $(LIBS) 

test_nn_prediction.exe: AtomicNetwork.o symmetric_functions.o network.o ensemble.o analyze_ensemble.o utils.o atoms.o matrix.o test_nn_prediction.cpp
	$(CPP) $(CFLAGS) AtomicNetwork.o symmetric_functions.o network.o ensemble.o atoms.o analyze_ensemble.o matrix.o utils.o test_nn_prediction.cpp -o test_nn_prediction.exe $(LIBS) $(GSLFLAGS) $(BOOSTFLAGS)

test_sd_deriv.exe: symmetric_functions.o atoms.o test_sf_deriv.cpp
	$(CPP) $(CFLAGS) atoms.o symmetric_functions.o test_sf_deriv.cpp -o test_sf_deriv.exe $(LIBS)

test_ensemble.exe: ensemble.o atoms.o test_ensemble.cpp symmetric_functions.o analyze_ensemble.o matrix.o
	$(CPP) $(CFLAGS) atoms.o matrix.o ensemble.o analyze_ensemble.o symmetric_functions.o test_ensemble.cpp -o test_ensemble.exe $(LIBS) $(GSLFLAGS) $(BOOSTFLAGS)

test_read_atoms.exe: atoms.o test_read_atoms.cpp
	$(CPP) $(CFLAGS) atoms.o test_read_atoms.cpp -o test_read_atoms.exe

AtomicNetwork.o: AtomicNetwork.hpp AtomicNetwork.cpp symmetric_functions.o
	$(CPP) $(CFLAGS) -c AtomicNetwork.cpp -o AtomicNetwork.o

analyze_ensemble.o: analyze_ensemble.cpp analyze_ensemble.hpp
	$(CPP) $(CFLAGS) -c analyze_ensemble.cpp -o analyze_ensemble.o

symmetric_functions.o: symmetric_functions.hpp symmetric_functions.cpp atoms.hpp
	$(CPP) -c symmetric_functions.cpp -o symmetric_functions.o

ensemble.o: ensemble.hpp ensemble.cpp
	$(CPP) -c ensemble.cpp -o ensemble.o

atoms.o: atoms.cpp atoms.hpp
	$(CPP) -c atoms.cpp -o atoms.o

network.o: network.cpp network.hpp
	$(CPP) -c network.cpp -o network.o

utils.o: utils.cpp utils.hpp
	$(CPP) -c utils.cpp -o utils.o

matrix.o: matrix.c matrix.h
	$(CPP) -c matrix.c -o matrix.o -fpermissive
