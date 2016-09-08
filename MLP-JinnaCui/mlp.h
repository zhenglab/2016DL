/*£°
*\  MLP
*\  MLP for Classification
*\  Jinna
*\  Jinna_ouc@163.com
*\  https://github.com/Jinnaouc
*\  2016.08.22
*/
#include <iostream>//±Í◊º ‰»Î ‰≥ˆ¡˜
#include <math.h>
#include <stdlib.h>  //for abs
#include <stdio.h> //±Í◊º ‰»Î ‰≥ˆ
#include <fstream>
using namespace std;

#define e 2.718281
#define data_number 299
#define data_dimension 30
#define input_layer_node_number 30
#define first_hiden_layer_node_number 40
#define second_hiden_layer_node_number 40
#define output_layer_node_number 2
#define test_data_number 200
#define learning_rate 0.1
#define epoch_num 50000
#define batch_size 10
//∂®“Â ‰»Î≤„µƒ±‰¡ø^^^^^^^^
class Inputlayer{
public:
	float input_data[data_number][data_dimension];//∂®“Â ‰»Î ˝æ›∂˛Œ¨ ˝◊È£ª
	float label[data_number];
	float input_test_data[test_data_number][data_dimension];

};

//∂®“Â√ø≤„Ω⁄µ„ ˝÷µ∫Õ“˛≤ÿ≤„Ω⁄µ„Ã›∂»¿€º”÷µ≤Œ ˝^^^^^^^^^^
class Node_value{
public:
	float first_hiden_layer_node_value[data_number][first_hiden_layer_node_number];
	float second_hiden_layer_node_value[data_number][second_hiden_layer_node_number];
	float output_layer_node_value[data_number][output_layer_node_number];
	float first_hiden_layer_node_grad_value[first_hiden_layer_node_number];
	float second_hiden_layer_node_grad_value[second_hiden_layer_node_number];
};

//∂®“Â»®÷ÿ∫Õ∆´÷√ ˝◊È≤¢≥ı ºªØ^^^^^^^
class Parameter{
public:
	float ih_weight_array[input_layer_node_number][first_hiden_layer_node_number];//∂®“Â ‰»Î≤„∫Õ“˛≤ÿ≤„µƒ∂˛Œ¨»®÷ÿ ˝◊È£ª
	float hh_weight_array[first_hiden_layer_node_number][second_hiden_layer_node_number];//∂®“Â“˛≤ÿ≤„÷Æº‰µƒ∂˛Œ¨»®÷ÿ ˝◊È£ª
	float ho_weight_array[second_hiden_layer_node_number][output_layer_node_number];//∂®“Â“˛≤ÿ≤„∫Õ ‰≥ˆ≤„µƒ∂˛Œ¨»®÷ÿ ˝◊È£ª
	float first_hiden_layer_bias_array[first_hiden_layer_node_number];//∂®“Âµ⁄“ª“˛≤ÿ≤„µƒ“ªŒ¨∆´÷√ ˝◊È£ª
	float second_hiden_layer_bias_array[second_hiden_layer_node_number];//∂®“Âµ⁄∂˛“˛≤ÿ≤„µƒ“ªŒ¨∆´÷√ ˝◊È£ª
	float output_layer_bias_array[output_layer_node_number];//∂®“Â ‰≥ˆ≤„µƒ“ªŒ¨∆´÷√ ˝◊È£ª
	float sample_from_gaussian(float miu, float sigma);//∂®“Â∏ﬂÀπ∑÷≤º»°£®0°´1£©µƒÀÊª˙÷µ£ª

};

//∂®“Â»®÷ÿ∫Õ∆´÷√µƒ”≈ªØæÿ’Û
class Gradparameter{
public:
	float ih_weight_grad_array[input_layer_node_number][first_hiden_layer_node_number];
	float hh_weight_grad_array[first_hiden_layer_node_number][second_hiden_layer_node_number];
	float ho_weight_grad_array[second_hiden_layer_node_number][output_layer_node_number];
	float first_hiden_layer_bias_grad_array[first_hiden_layer_node_number];
	float second_hiden_layer_bias_grad_array[second_hiden_layer_node_number];
	float output_layer_bias_grad_array[output_layer_node_number];
};

//∂®“Â∏˜∏ˆ∫Ø ˝^^^^^^^^^^^^^^
class Functions{
public:
	float sigmoid(float x);
    float gradsigmoid(float x);
	float error();
};