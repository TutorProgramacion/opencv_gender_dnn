
cmake_minimum_required(VERSION 3.13)                 
PROJECT(gender)                                      

FIND_PACKAGE( OpenCV REQUIRED )                     
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(gender main.cpp)                          
TARGET_LINK_LIBRARIES (gender ${OpenCV_LIBS})         

file(COPY caffe DESTINATION caffe-models)