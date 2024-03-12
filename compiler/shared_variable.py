class SharedVariableContainer:
    # 生成模式有两种，
    # [0,层数]：测单层，用于单层仿真,生成单层指令、权重、对比结果
    # [1,层数]：联测,用于侧板子,生成1-本层的连续指令、权重、对比结果
    generate_mode = [1, 143]
    # 这三个变量需要配合生成模式使用
    gen_ins = True  # 是否需要指令
    gen_weight = False  # 是否需要生成权重
    gen_result = False  # 是否需要生成对比结果,联测生成一次后建议关掉,太占用时间

    picture_address = 0  # 输入图片地址,一般记0
    write_address = 0x640000  # 特征图起始地址,一般接在输入图片之后
    weight_address = 0x5C17000  # 权重起始地址,一般接在所有特征图之后

    img_size = 640  # 输入图片规定的尺寸
    parallel = 16  # 指示生成数据采用16进16出还是8进8出
    img_path = '../yolo_obb/image/17942.bmp'  # 输入图片路径
    model_path = '../yolo_obb/Net-16-quantization_post.pth'  # 量化模型路径
    mean = 0  # 整个数据集图片的均值,无均值填0
    std = 1  # 整个数据集图片的方差,无方差填1
    file_path = "../sim_data/"  # 生成指令、权重、对比结果路径
    para_path = "../yolo_obb/para_last/"  # 提取npy文件路径
    start_op = 1  # 用于首次卷积时指示输入是否要做特殊处理
    layer_count = 1  # 用于维护网络层数
    # shape状态字典,用于维护shape操作的状态码
    shape_control = {
        'MaxPool': 1,
        'UpSample': 2,
        'Concat': 3,
        'Split': 4,
        'Add': 5,
        'LeakyRelu': 6,
        'MeanPool': 7,
        'Mul': 8,
        'ArgMax': 9,
        'UpBili': 10,
        'Pre': 11,
        'YoloSig': 12
    }
    address_table = []  # 读地址字典,用于维护特征图读地址变换
    layer_table = {}  # 特征图字典,记录网络结构每一层对应的特征图id
