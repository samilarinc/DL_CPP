all:
	clear
	g++ -O3 -Wall -shared -fPIC -I/home/codespace/.python/current/include/python3.10 -I/home/codespace/.python/current/lib/python3.10/site-packages/pybind11/include \
	-I /workspaces/DL_CPP/Layers 		-I  /workspaces/DL_CPP/Loss \
	-I /workspaces/DL_CPP/Optimizers 	-I  /workspaces/DL_CPP/Tensor \
	-I /workspaces/DL_CPP/Layers 	 	-I 	/workspaces/DL_CPP \
	-I /workspaces/DL_CPP/Activations \
	Tensor/Tensor.cpp Loss/L1Loss.cpp Loss/L2Loss.cpp Loss/CrossEntropyLoss.cpp \
	Optimizers/SGD.cpp Optimizers/Adam.cpp Layers/Dense.cpp Activations/ReLU.cpp Activations/Sigmoid.cpp \
	Layers/Dropout.cpp Layers/L1_Regularizer.cpp Layers/L2_Regularizer.cpp \
	bind_config.cpp -o pyflow.cpython-310-x86_64-linux-gnu.so
clean:
	rm -f *.o *.so