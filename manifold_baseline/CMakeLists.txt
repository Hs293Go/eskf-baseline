cmake_minimum_required(VERSION 3.28.0)
project(manifold_baseline VERSION 0.1.0)

add_library(manifold_baseline_manifold_baseline INTERFACE)
target_sources(
  manifold_baseline_manifold_baseline
  INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
            include/manifold_baseline/manifold_baseline.hpp)
target_compile_features(manifold_baseline_manifold_baseline
                        INTERFACE cxx_std_20)
add_library(manifold_baseline::manifold_baseline ALIAS
            manifold_baseline_manifold_baseline)
