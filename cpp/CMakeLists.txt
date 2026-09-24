cmake_minimum_required(VERSION 3.16)
project(sdf_gradients LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module"                  ON)
option(USE_OPENMP            "Enable OpenMP parallelism"                      ON)
option(SDF_WITH_LBFGSPP      "Build the LBFGS++ reference gradient optimizer" ON)

# CPU tuning — applied globally so sdf_core and sdf_cpp share the same Eigen
# vectorization. Tuning only one target causes an Eigen ABI mismatch across
# TU boundaries → segfault in PUInterpolator::fit.
#
# ``native`` suits building and running on the same machine. For a binary
# that must also run elsewhere, pick the oldest target CPU instead:
#   x86 clusters (login node with more ISA features than some compute nodes):
#     cmake -DSDF_MARCH=x86-64-v3 ..   (AVX2 + FMA3 + BMI2)
#   Apple Silicon, any M-series chip:
#     cmake -DSDF_MCPU=apple-m1 ..
set(SDF_MARCH "native" CACHE STRING
    "Compiler -march flag (use 'x86-64-v3' for cluster-portable builds)")
set(SDF_MCPU "native" CACHE STRING
    "Compiler -mcpu flag on Apple Silicon (use 'apple-m1' to run on any M-series)")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_compile_options(-mcpu=${SDF_MCPU})
elseif(NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=${SDF_MARCH})
endif()

include(FetchContent)

# ── Eigen3 ───────────────────────────────────────────────────────────
# System Eigen (brew / apt) if present, otherwise fetch 3.4.0.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found — fetching 3.4.0")
    set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING         OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(eigen)
endif()

# ── CGAL (+ Boost headers, GMP/MPFR) ─────────────────────────────────
# Required: the sphere-neighbor search in get_visible_arcs goes through the
# power diagram in sphere_intersect_core.cpp. Header-only for our usage.
find_path(BOOST_INCLUDE_DIR boost/config.hpp
    HINTS /opt/homebrew/include /usr/local/include /usr/include)
if(NOT BOOST_INCLUDE_DIR)
    message(STATUS "Boost headers not found — fetching 1.84.0 (headers only, ~120MB)")
    FetchContent_Declare(boost
        URL      https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.bz2
        URL_HASH SHA256=cc4b893acf645c9d4b698e9a0f08ca8846aa5d6c68275c14c3e7949c24109454)
    FetchContent_Populate(boost)
    set(BOOST_INCLUDE_DIR "${boost_SOURCE_DIR}")
endif()

find_path(CGAL_INCLUDE_DIR CGAL/Regular_triangulation_3.h
    HINTS /opt/homebrew/include /usr/local/include /usr/include)
if(NOT CGAL_INCLUDE_DIR)
    message(STATUS "CGAL headers not found — fetching v5.6.2 (large download)")
    FetchContent_Declare(cgal
        GIT_REPOSITORY https://github.com/CGAL/cgal.git
        GIT_TAG        v5.6.2
        GIT_SHALLOW    TRUE)
    FetchContent_Populate(cgal)
    # The CGAL git repo uses a per-package layout: <Package>/include/CGAL/...
    file(GLOB CGAL_INCLUDE_DIR LIST_DIRECTORIES true "${cgal_SOURCE_DIR}/*/include")
endif()

# GMP + MPFR speed up CGAL's exact fallback but are not required: without them
# CGAL uses Boost.Multiprecision, so sudo-less machines need nothing extra.
find_library(GMP_LIBRARY  NAMES gmp
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(MPFR_LIBRARY NAMES mpfr
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

# ── Header-only libraries ────────────────────────────────────────────
# nanoflann is vendored under third_party/.
set(NANOFLANN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann/include)

# libigl: only for igl/marching_cubes.h.
FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    TRUE)
FetchContent_GetProperties(libigl)
if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
endif()

# LBFGS++: only for the reference optimize_best_gradients (grad_optimizer
# "lbfgspp") that the batched in-house BFGS is benchmarked against.
if(SDF_WITH_LBFGSPP)
    FetchContent_Declare(lbfgspp
        GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
        GIT_TAG        v0.4.0
        GIT_SHALLOW    TRUE)
    FetchContent_GetProperties(lbfgspp)
    if(NOT lbfgspp_POPULATED)
        FetchContent_Populate(lbfgspp)
    endif()
endif()

# ── OpenMP ───────────────────────────────────────────────────────────
if(USE_OPENMP)
    # Apple clang ships without OpenMP; point FindOpenMP at Homebrew's libomp.
    if(APPLE AND NOT OpenMP_CXX_FLAGS)
        execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES omp)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        endif()
    endif()
    find_package(OpenMP)
endif()

# ── pybind11 ─────────────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    # Pick one interpreter and build the module for it. FindPython honors
    # -DPython_EXECUTABLE (or the legacy -DPYTHON_EXECUTABLE) and otherwise
    # prefers the active conda env / virtualenv over whatever `python3` comes
    # first on PATH, so the .so matches the Python that will import it.
    if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
        set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    endif()
    if(CMAKE_VERSION VERSION_LESS 3.18)
        find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
    else()
        find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
    endif()
    message(STATUS "Python bindings for ${Python_EXECUTABLE} (${Python_VERSION})")
    set(PYBIND11_FINDPYTHON ON)  # pybind11 reuses the Python found above

    # Prefer the pybind11 installed in that Python, then a system one, then fetch.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    find_package(pybind11 CONFIG QUIET HINTS "${_pybind11_dir}")
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found — fetching v2.13.6")
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ── Core static library ──────────────────────────────────────────────
add_library(sdf_core STATIC
    src/interpolator.cpp
    src/optimize_lbfgspp.cpp
    src/duchon_interpolator.cpp
    src/pu_interpolator.cpp
    src/visibility.cpp
    src/clamp.cpp
    src/optimization.cpp
    src/main_algorithm.cpp
    src/get_visible_arcs.cpp
    src/sphere_intersect_core.cpp
    src/sphere_bvh.cpp
    src/existing_modules_adapter.cpp
)

target_include_directories(sdf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(sdf_core SYSTEM PUBLIC
    ${NANOFLANN_INCLUDE_DIR}
    ${libigl_SOURCE_DIR}/include
    ${CGAL_INCLUDE_DIR}
    ${BOOST_INCLUDE_DIR})
target_link_libraries(sdf_core PUBLIC Eigen3::Eigen)
target_compile_definitions(sdf_core PUBLIC HAVE_CGAL)

if(GMP_LIBRARY AND MPFR_LIBRARY)
    target_link_libraries(sdf_core PUBLIC ${MPFR_LIBRARY} ${GMP_LIBRARY})
else()
    message(STATUS "GMP/MPFR not found — CGAL falls back to Boost.Multiprecision")
    target_compile_definitions(sdf_core PUBLIC CGAL_DISABLE_GMP)
endif()

if(SDF_WITH_LBFGSPP)
    target_include_directories(sdf_core SYSTEM PUBLIC ${lbfgspp_SOURCE_DIR}/include)
    target_compile_definitions(sdf_core PUBLIC SDF_HAVE_LBFGSPP)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(sdf_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(sdf_core PUBLIC USE_OPENMP)
endif()

# Accelerate on macOS. On Linux, OpenBLAS is deliberately not linked: the hot
# path is PU patches of 100–700 rows, where Eigen's own vectorized kernels beat
# BLAS dispatch overhead (linking it measurably regressed runtime).
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(sdf_core PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(sdf_core PUBLIC EIGEN_USE_BLAS)
    endif()
endif()

# ── Python bindings ──────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(sdf_cpp bindings/pybind_main.cpp)
    target_link_libraries(sdf_cpp PRIVATE sdf_core)
endif()
