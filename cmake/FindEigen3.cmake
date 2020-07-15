FIND_PATH( EIGEN3_INCLUDE_DIRS Eigen/Geometry
    $ENV{EIGEN3DIR}/include
    /usr/local/include/eigen3
    /usr/local/include
    /usr/local/X11R6/include
    /usr/X11R6/include
    /usr/X11/include
    /usr/include/X11
    /usr/include/eigen3/
    /usr/include
    /opt/X11/include
    /opt/include 
    ${CMAKE_SOURCE_DIR}/external/eigen/include)

SET(EIGEN3_FOUND "NO")
IF(EIGEN3_INCLUDE_DIRS)
    SET(EIGEN3_FOUND "YES")
ENDIF()
