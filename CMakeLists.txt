cmake_minimum_required(VERSION 3.18)
project(parallel_sentiment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenMP flags for macOS
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(preprocessor 
    preprocessor.cpp
    one_hot_embedder.cpp
    main.cpp)

# Link libraries
target_link_libraries(preprocessor 
    PRIVATE 
    OpenMP::OpenMP_CXX)

# Include directories
target_include_directories(preprocessor 
    PRIVATE 
    "${CMAKE_SOURCE_DIR}"
    "/opt/homebrew/opt/libomp/include")