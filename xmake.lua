add_rules("mode.debug", "mode.release")

add_requires("openmp")
add_requires("openblas")
add_requires("fmt")

target("benchmark_cpu")
    set_kind("binary")
    set_languages("cxx11")
    add_vectorexts("avx")
    add_cxflags("/openmp:experimental",{toolchain="msvc"})
    add_files({
        "benchmark_cpu.cpp",
        "matrixMultiply.cpp"}
    )
    add_packages("openmp")
    add_packages("openblas")
    add_packages("fmt")