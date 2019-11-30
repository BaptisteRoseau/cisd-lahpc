debug:
	cmake -H. -Bunix-builds/Debug -DCMAKE_BUILD_TYPE=Debug "-GUnix Makefiles" && cd unix-builds/Debug && make
