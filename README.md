# Advanced Game Physics (AGPhys) WS 20/21


## Installation unter Linux

Hier sind die Abhängigkeiten und deren Installationen beschrieben.
Das ganze wurde getestet unter Ubuntu 20.04 und mit der CUDA Version 11.4.

### CUDA 

Stellen Sie sicher, dass Sie einen ausreichend aktuellen Grafiktreiber installiert haben.
Falls nicht, laden Sie diesen bei Nvidia runter oder installieren Sie ihn über die NVIDIA PPA.

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt install nvidia-driver-455
    sudo reboot
    
Installieren Sie nun das CUDA Toolkit. https://developer.nvidia.com/cuda-downloads

| CUDA Toolkit | Driver Version | Min Architecture | Max. GCC Version |
|:----:|:----:|:----:|:----:|
| CUDA 11.1 (recommended) | >= 455 | Maxwell (GTX 9xx) | 9 | 
| CUDA 11.0| >= 450 | Maxwell (GTX 9xx) | 9 | 
| CUDA 10.2 | >= 440 | Kepler (GTX 6xx) | 8 | 

    # Example for Cuda 11.4
    wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda_11.4.2_470.57.02_linux.run
    sudo sh cuda_11.4.2_470.57.02_linux.run
    
### CMake

Installiere über snap oder apt:

    sudo snap install cmake --classic
    # Make sure that the Version is min 3.18
    cmake --version
    
Oder einfach selber bauen:

    wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz
    tar -zxvf cmake-3.18.4.tar.gz
    cd cmake-3.18.4
    ./bootstrap
    make
    sudo make install
    
    
### Agphys 

    git clone git@git9.cs.fau.de:agphys21/$Benutzerkennung.git agphys
    cd agphys

    # Saiga und weitere dependencies klonen
    git submodule update --init --recursive

    # Und bauen
    mkdir build
    cd build
    
    # Falls CUDA 10.2 verwendet wird:
    export CXX=g++-8
    export CUDAHOSTCXX=g++-8
    
    # Falls CUDA 11.1 verwendet wird:
    export CXX=g++-9
    export CUDAHOSTCXX=g++-9

    cmake ..
    # CMake muss in nur bei Aenderungen in der CMakeLists.txt erneut 
    # ausgefuehrt werden, ansonsten reicht ab sofort ein make
    make -j8

    # Und letztendlich auch ausfuehren :)
    cd ..
    ./agphys
