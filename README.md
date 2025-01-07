# 算子库

跨平台高性能通用通信算子库。形式为 C 接口动态库。

## 一、使用说明

### 依赖安装

#### Open MPI库
编译CPU版本库需要安装Open MPI，并配置Open MPI的路径为 `MPI_HOME `。

### 配置

#### 查看当前配置

```xmake
xmake f -v
```

#### 配置 CPU （默认配置）

```xmake
xmake f --cpu=true -cv
```

#### 配置 GPU

需要指定 CUDA 路径， 一般为 `CUDA_HOME` 或者 `CUDA_ROOT`。

```xmake
xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv
```

#### 配置 MLU

```xmake
xmake f --cambricon-mlu=true -cv
```

### 编译

```xmake
xmake
```

### 将编译好的算子库添加至环境变量 `INFINI_ROOT`

```bash
export INFINI_ROOT=[PATH_TO_LIBRARY]
```

### 运行算子测试

```bash
cd operatorspy/tests
python operator_name.py
```

## 二、开发说明

### 目录结构

```bash
├── xmake.lua  # xmake 构建脚本
├── include
│   ├── ops
│   │   ├── [operator_name].h  # 对外暴露的算子 C 接口定义，descriptor 定义
│   ├── tensor
│   │   ├── tensor_descriptor.h  # 对外暴露的张量 descriptor 定义
│   ├── *.h  # 对外暴露的核心结构体定义
├── src
│   ├── devices
│   │   ├── [device_name]
│   │       ├── *.cc/.h # 特定硬件（如 cpu、英伟达）通用代码
│   ├── ops
│   │   ├── utils.h  # 全算子通用代码 (如 assert)
│   │   ├── [operator_name]  # 算子实现目录
│   │       ├── operator.cc # 算子 C 接口实现 (根据 descriptor 调用不同的算子实现)
│   │       ├── [device_name]
│   │       │   ├── *.cc/.h/... # 特定硬件的算子实现代码
│   ├── *.h  # 核心结构体定义
│  
├── operatorspy  # Python 封装以及测试脚本
    ├── tests
    │   ├── operator_name.py  # 测试脚本
    ├── *.py     # Python 封装代码
```

### 增加新的硬件

- 在 `src/device.h` 和 `operatorspy/devices.py` 中增加新的硬件类型，注意两者需要一一对应；
- 在 `xmake.lua` 中增加新硬件的编译选项以及编译方式；
- 在 `src/ops/devices/[device_name]` 下编写特定硬件的通用代码；
- 实现该硬件的算子；

### 增加新的算子

- 在 `src/ops/[operator_name]` 增加创建/销毁算子描述符、算子计算的C接口，注意C接口header使用`__C __export`前缀；
- 在 `src/ops/[operator_name]/[device_name]` 增加算子在各硬件的实现代码；
- 在 `operatorspy/tests/[operator_name].py` 增加算子测试；
