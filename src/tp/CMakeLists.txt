add_executable( kmeans kmeans.cpp ocv_utils.cpp ocv_utils.hpp)
target_link_libraries( kmeans ${OpenCV_LIBS} )

add_executable( meanshift meanshift.cpp ocv_utils.cpp ocv_utils.hpp)
target_link_libraries( meanshift ${OpenCV_LIBS} )