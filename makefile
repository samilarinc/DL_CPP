all:
	g++ -fPIC -shared -I/usr/include/python3.8 -I/home/samil/.local/lib/python3.8/site-packages/pybind11/include \
	-I /mnt/c/Users/pc/repos/DL_Framework_C/Layers -I /mnt/c/Users/pc/repos/DL_Framework_C/Loss \
	-I /mnt/c/Users/pc/repos/DL_Framework_C/Optimizers -I /mnt/c/Users/pc/repos/DL_Framework_C/Tensor \
	-I /mnt/c/Users/pc/repos/DL_Framework_C/Layers -I /mnt/c/Users/pc/repos/DL_Framework_C \
	Tensor/Tensor.cpp Loss/L1Loss.cpp Loss/L2Loss.cpp Loss/CrossEntropyLoss.cpp \
	Optimizers/SGD.cpp Layers/Dense.cpp bind_config.cpp -o pyflow.cpython-38-x86_64-linux-gnu.so
clean:
	rm -f *.o *.so