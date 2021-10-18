
#include "saiga/core/framework/framework.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/util/crash.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/tests/test.h"

#include "agphys.h"

#include <iostream>


#if !defined(SAIGA_USE_GLFW) || !defined(SAIGA_USE_OPENGL)
#    error Missing Saiga dependencies.
#endif

int main(int argc, char* args[])
{
    Saiga::CUDA::initCUDA();
    Saiga::CUDA::printCUDAInfo();

    Saiga::catchSegFaults();

    Saiga::EigenHelper::EigenCompileFlags flags;
    flags.create<2954617>();
    std::cout << flags << std::endl;
    Saiga::EigenHelper::checkEigenCompabitilty<23598615>();

    {
        Agphys window;
        window.run();
    }

    Saiga::CUDA::destroyCUDA();
    std::cout << "Done." << std::endl;
    return 0;
}
