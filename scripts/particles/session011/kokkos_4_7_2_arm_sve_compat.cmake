# Kokkos 4.7.2 uses check_include_file_cxx() in its ARM SVE SIMD setup
# without loading the CMake module that defines the command.
include(CheckIncludeFileCXX)
