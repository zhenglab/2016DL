/*！
 *\  MLP
 *\  MLP for Classification
 *\  Jinna
 *\  Jinna_ouc@163.com
 *\  https://github.com/Jinnaouc
 *\  2016.08.22
 */
#include <iostream>//标准输入输出流
#include <math.h>
#include <stdio.h> //标准输入输出
#include <stdlib.h> //宏、通用函数
#include <fstream>
using namespace std;

#define e 2.718281
#define data_number 300
#define data_dimension 30
#define input_layer_node_number 30
#define first_hiden_layer_node_number 20
#define second_hiden_layer_node_number 10
#define output_layer_node_number 2
#define learning_rate 0.01

//定义输入层的变量^^^^^^^^
class Inputlayer{
public:
    float input_data[data_number][data_dimension];//定义输入数据二维数组；
    float label[data_number];
};

//定义每层节点数值和隐藏层节点梯度累加值参数^^^^^^^^^^
class Node_value{
public:
    float first_hiden_layer_node_value[first_hiden_layer_node_number];
    float second_hiden_layer_node_value [second_hiden_layer_node_number];
    float output_layer_node_value[output_layer_node_number];
    float first_hiden_layer_node_grad_value[first_hiden_layer_node_number];
    float second_hiden_layer_node_grad_value [second_hiden_layer_node_number];
};

//定义权重和偏置数组并初始化^^^^^^^
class Parameter{
public:
    float ih_weight_array[input_layer_node_number][first_hiden_layer_node_number];//定义输入层和隐藏层的二维权重数组；
    float hh_weight_array[first_hiden_layer_node_number][second_hiden_layer_node_number];//定义隐藏层之间的二维权重数组；
    float ho_weight_array[second_hiden_layer_node_number][output_layer_node_number];//定义隐藏层和输出层的二维权重数组；
    float first_hiden_layer_bias_array[first_hiden_layer_node_number];//定义第一隐藏层的一维偏置数组；
    float second_hiden_layer_bias_array[second_hiden_layer_node_number];//定义第二隐藏层的一维偏置数组；
    float output_layer_bias_array[output_layer_node_number];//定义输出层的一维偏置数组；
    float sample_from_gaussian(float miu,float sigma);//定义高斯分布取（0～1）的随机值；
};

//定义权重和偏置的优化矩阵
class Gradparameter{
public:
    float ih_weight_grad_array[input_layer_node_number][first_hiden_layer_node_number];
    float hh_weight_grad_array[first_hiden_layer_node_number][second_hiden_layer_node_number];
    float ho_weight_grad_array[second_hiden_layer_node_number][output_layer_node_number];
    float first_hiden_layer_bias_grad_array[first_hiden_layer_node_number];
    float second_hiden_layer_bias_grad_array[second_hiden_layer_node_number];
    float output_layer_bias_grad_array[output_layer_node_number];
};

//定义各个函数^^^^^^^^^^^^^^
class Functions{
public:
    float sigmoid(float x);
    float gradsigmoid(float x);
    float error();
};
















