FIND_PATH(
    # filled variable
    NANOFLANN_INCLUDE_DIR
    # what I am looking for?
    nanoflann.hpp
    # where should I look?
    $ENV{NANOFLANN_DIR}
    ${CMAKE_SOURCE_DIR}/include)

IF(NANOFLANN_INCLUDE_DIR)
   SET(NANOFLANN_FOUND TRUE)
else()
   SET(NANOFLANN_FOUND FALSE)
ENDIF()

IF(NANOFLANN_FOUND)
    IF(NOT CMAKE_FIND_QUIETLY)
        MESSAGE(STATUS "Found NanoFlann: ${NANOFLANN_INCLUDE_DIR}")
    ENDIF()
ELSE()
    IF(NANOFLANN_FOUND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find NanoFlann")
    ENDIF()
ENDIF()
