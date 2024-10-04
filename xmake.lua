add_rules("mode.debug", "mode.release")

set_config("blas","mkl")
-- set_config("blas","openblas")

add_requires("openmp")
add_requires("openblas")
add_requires("fmt")
add_requires("pthreads4w")

target("benchmark_cpu")
    set_kind("binary")
    set_languages("cxx11")
    add_vectorexts("avx")
    add_cxflags("/openmp:experimental",{toolchain="msvc"})
    add_cxflags("/W4 ",{toolchain="msvc"})
    add_files({
        "benchmark_cpu.cpp",
        "matrixMultiply.cpp"}
    )
    add_packages("openmp")
    add_packages("fmt")
    add_packages("pthreads4w")
    if(get_config("blas")=="openblas")then
        add_defines("BLAS_OPENBLAS")
        add_packages("openblas")
    elseif (get_config("blas")=="mkl") then 
        add_defines("BLAS_MKL")
        add_includedirs("C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/include")
        add_linkdirs("C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/lib")
        add_linkdirs("C:/Program Files (x86)/Intel/oneAPI/tbb/2021.13/lib")
        add_links("mkl_intel_lp64_dll","mkl_core_dll")
    end     


    