* 环境
  ubuntu 14.04
  g++-4.9
  
* 编译

  首先进入到 ch_mb 目录
  
  mkdir build
  cd build

  cmake ..
  make
  make install 

* 测试

  ./test_main.bin
  
* 说明

  1. 执行 make install 后,头文件会安装在 build/include 目录, 库文件安装在 build/lib 目录.
  2. test_main.cpp : 测试接口函数源文件
  3. CMSS_FaceRecognize.cpp : 接口函数源文件, 将被编译为 libface_recognition.so 库
  4. 接口函数参数说明在 CMSS_FaceRecognize.h 中
