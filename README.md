# 使用指南

此文档为开发平台竞赛示例代码中SDK封装部分的使用说明。

## 用途

在线上完成模型训练之后，后续步骤即为模型测试，而为了能够使用线上平台的自动测试功能，需要将模型封装成统一的接口。这份示例代码提供了两种可选的SDK封装方式：

1. 使用C++语言封装模型；
2. 使用Python语言封装模型。

## 目录

### 代码目录结构

```shell
ev_sdk
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- include         # 头文件目录
|   `-- ji.h        # 需要实现的接口头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- src             # ji.h的接口实现代码，如ji.cpp、ji.py
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```
## 示例代码的使用方法

示例代码使用一个**ssd_inception_v2**模型的行人检测算法，推理引擎为Tensorflow。在运行**示例代码**的测试程序之前，需要手动下载预训练的模型文件：

```shell
wget http://10.9.0.146:8888/group1/M00/00/01/CgkA616-GFeEZMcNAAAAAHEKBs82984.gz -O /tmp/ssd_inception_v2_pedestrian.tar.gz && tar zxf /tmp/ssd_inception_v2_pedestrian.tar.gz -C /usr/local/ev_sdk/model
```

示例代码可以直接拷贝到`/usr/local/ev_sdk/`下提交并发起测试。

## 如何基于示例代码封装自己的SDK

### 使用C++接口

根据自己实际的模型名称、模型输入输出、模型推理逻辑，修改`src/SampleDetectorImpl.cpp`

1. 实现模型初始化：

   ```c++
   int SampleDetector::init();
   ```

2. 实现模型推理：

   ```c++
   STATUS SampleDetector::processImage(const cv::Mat &cv_image, std::vector<Object> &result);
   ```

3. 根据实际项目，将结果封装成项目所规定的输入输出格式

   示例代码中使用的是目标检测类项目，因此需要根据实际项目，添加检测类别信息：

   ```c++
   # src/SampleDetectorImpl.cpp, SampleDetector::init()
   mIDNameMap.insert(std::make_pair<int, std::string>(1, "class0"));	// id 1, 类别名称 class0
   mIDNameMap.insert(std::make_pair<int, std::string>(2, "class1")); // id 2, 类别名称 class1
   ```

4. 编译程序

   ```shell
   mkdir -p /usr/local/ev_sdk/build
   cd /usr/local/ev_sdk/build
   make install
   ```

5. 运行测试程序

   ```shell
   cd /usr/local/ev_sdk/bin
   ./test-ji-api -f 1 -i ../data/dog.jpg
   ```

6. 样例输出结果

   ```shell
   {
     "objects":	[{
         "xmin":	116,
         "ymin":	557,
         "xmax":	557,
         "ymax":	860,
         "confidence":	0.988156,
         "name":	"class1"
       }]
   }
   ```

### 使用Python接口

与C++接口不同，当使用Python接口发起测试时，系统仅会运行`src/ji.py`内的代码，用户需要根据自己的模型名称、模型输入输出、模型推理逻辑，修改`src/ji.py`

1. 实现模型初始化：

   ```python
   # src/ji.py
   def init()
   ```

2. 实现模型推理：

   ```python
   # src/ji.py
   def process_image(net, input_image)
   ```
   其中process_image接口返回值，必须是JSON格式的字符串，并且格式符合要求。

3. 根据实际项目，将结果封装成项目所规定的输入输出格式

   示例代码中使用的是目标检测类项目，因此需要根据实际项目，添加检测类别信息：

   ```python
   # src/ji.py
   label_id_map = {1: "class0", 2: "class1"}
   ```
   
4. 测试程序

   ```shell
   python ji.py
   ```

5. 样例输出结果：

  ```json
  code: 0
  json: {
    "objects":	[{
        "xmin":	116,
        "ymin":	557,
        "xmax":	557,
        "ymax":	860,
        "confidence":	0.988156,
        "name":	"class1"
      }]
  }
  ```

6. 使用Python接口发起测试时，需要在`Dockerfile`中设置一个环境变量：

   ```dockerfile
   ENV AUTO_TEST_USE_JI_PYTHON_API=1
   ```