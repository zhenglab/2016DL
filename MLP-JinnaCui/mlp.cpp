#include "mlp.h"

//定义高斯随机数(仿写俞老师deep.cpp中的高斯随机数函数)
float Parameter::sample_from_gaussian(float miu,float sigma){
    static float V1,V2,S;
    static int phase = 0;
    float X;
    float gaussian_output;
    if (phase == 0){
        do{
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        }while (S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    gaussian_output=X * sigma + miu;
    return gaussian_output;
}

//定义激活函数sigmoid
float Functions::sigmoid(float x){
    float Y;
    Y=1/(1+exp(-x));
    return Y;
}

//定义sigmoid函数的导函数
float Functions::gradsigmoid(float x){
    float G;
    G=x*(1-x);
    return G;
}
//定义前馈网络^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
float feedforward(Inputlayer &inp,Parameter &par){
    Node_value nva;
    Functions fun;
    float a,b,c,sum;
    float loss;
    a = 0;
    
    //通过循环求解每层节点的值
    for (int i=0;i<first_hiden_layer_node_number;i++) {//第一隐藏层节点输出值；
        for (int j=0; j<input_layer_node_number;j++) {
            a+=par.ih_weight_array[j][i]*inp.input_data[0][j];
        }
        nva.first_hiden_layer_node_value[i]=fun.sigmoid(a+par.first_hiden_layer_bias_array[i]);
                    cout<<"fhlayer_node_value"<<nva.first_hiden_layer_node_grad_value[i]<<endl;
    }
    for (int i=0;i<first_hiden_layer_node_number;i++) {//第一隐藏层节点输出值；
            }
    for (int i=0;i<second_hiden_layer_node_number;i++) {//第二隐藏层节点输出值；
        for (int j=0; j<first_hiden_layer_node_number;j++) {
            b+=par.hh_weight_array[j][i]*nva.first_hiden_layer_node_value[j];
        }
        nva.second_hiden_layer_node_value[i]=fun.sigmoid(b+par.second_hiden_layer_bias_array[i]);
    }
    for (int i=0;i<output_layer_node_number;i++) {//输出层节点累加值；
        for (int j=0; j<second_hiden_layer_node_number;j++) {
            c+=par.ho_weight_array[j][i]*nva.second_hiden_layer_node_value[j];
        }
        sum+=exp(c+par.output_layer_bias_array[i]);
    }
    for (int i=0;i<output_layer_node_number;i++) {//输出层节点经softmax函数后的输出值；
        for (int j=0; j<second_hiden_layer_node_number;j++) {
            c+=par.ho_weight_array[j][i]*nva.second_hiden_layer_node_value[j];
        }
        nva.output_layer_node_value[i]=exp(c+par.output_layer_bias_array[i])/sum;
    }
    return 0;
}

//定义反馈网络^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
float feedback(Node_value &nva,Parameter &par){
    Gradparameter gpr;
    Inputlayer inp;
    Functions fun;
    
    //通过循环求解梯度并对权重和偏置进行优化**************
    //求解第二隐藏层到输出层权重梯度并优化权重；
    for (int i=0;i<second_hiden_layer_node_number;i++) {
        for (int j=0;j<output_layer_node_number;j++) {
            gpr.ho_weight_grad_array[i][j]=(nva.output_layer_node_value[j]-inp.label[j])*nva.second_hiden_layer_node_value[i];
            par.ho_weight_array[i][j]-=learning_rate*gpr.ho_weight_grad_array[i][j];
        }
    }
    //求解输出层的偏置梯度并对偏置优化偏置；
    for (int i=0;i<output_layer_node_number;i++) {
        gpr.output_layer_bias_grad_array[i]=nva.output_layer_node_value[i]-inp.label[i];
        par.output_layer_bias_array[i]-=learning_rate*gpr.output_layer_bias_grad_array[i];
    }
    //通过循环累加来定义输出层误差函数向第二隐藏层的传递；
    for (int i=0;i<second_hiden_layer_node_number;i++) {
        for (int j=0;j<output_layer_node_number;j++) {
            nva.second_hiden_layer_node_grad_value[i]+=(nva.output_layer_node_value[j]-inp.label[j])*par.ho_weight_array[i][j];
        }
    }
    //求第一隐藏层到第二隐藏层权重梯度并优化；
    for (int i=0;i<first_hiden_layer_node_number;i++) {
        for (int j=0;j<second_hiden_layer_node_number;j++) {
            gpr.hh_weight_grad_array[i][j]=nva.second_hiden_layer_node_grad_value[j]*fun.gradsigmoid(nva.second_hiden_layer_node_value[j])*nva.first_hiden_layer_node_value[i];
            par.hh_weight_array[i][j]-=learning_rate*gpr.hh_weight_grad_array[i][j];
        }
    }
    //求第二隐藏层的梯度偏置梯度并优化；
    for (int i=0;i<second_hiden_layer_node_number;i++) {
        gpr.second_hiden_layer_bias_grad_array[i]=nva.second_hiden_layer_node_grad_value[i]*fun.gradsigmoid(nva.second_hiden_layer_node_value[i]);
        par.second_hiden_layer_bias_array[i]-=learning_rate*gpr.second_hiden_layer_bias_grad_array[i];
    }
    //通过循环累加来定义第二隐藏层误差函数向第一隐藏层的传递；
    for (int i=0;i<first_hiden_layer_node_number;i++) {
        for (int j=0;j<second_hiden_layer_node_number;j++) {
            nva.first_hiden_layer_node_grad_value[i]+=nva.second_hiden_layer_node_grad_value[j]*fun.gradsigmoid(nva.second_hiden_layer_node_value[j])*par.hh_weight_array[i][j];
        }
    }
    //求输入层到第一隐藏层的权重梯度并优化；
    for (int i=0;i<input_layer_node_number;i++) {
        for (int j=0;j<first_hiden_layer_node_number;j++) {
            gpr.ih_weight_grad_array[i][j]=nva.first_hiden_layer_node_grad_value[j]*fun.gradsigmoid(nva.first_hiden_layer_node_value[j])*inp.input_data[0][i];
            par.ih_weight_array[i][j]-=learning_rate*gpr.ih_weight_grad_array[i][j];
        }
    }
    //求第一隐藏层偏置梯度并优化；
    for (int i=0;i<first_hiden_layer_node_number;i++) {
        gpr.first_hiden_layer_bias_grad_array[i]=nva.first_hiden_layer_node_grad_value[i]*fun.gradsigmoid(nva.first_hiden_layer_node_value[i]);
        par.first_hiden_layer_bias_array[i]-=learning_rate*gpr.first_hiden_layer_bias_grad_array[i];
    }
  
    return 0;
}


/************************************************************************************
                                  main function
 ************************************************************************************/
int main()
{
    Inputlayer inp;
    Parameter par;
     Node_value nva;
    //读入数据到input数组；
    ifstream data_file;
    data_file.open("data.txt");
    for(int i=0;i<data_number;i++)
    {
        for (int j=0;j<data_dimension;j++)
        {
            data_file>>inp.input_data[i][j];
        }
    }
    data_file.close();
    
    //读入label到数组
    ifstream label_file;
    label_file.open("datalabel.txt");
    for (int i=0 ;i<data_number;i++) {
        label_file>>inp.label[i];
    }
    label_file.close();
    
    //给权重、偏置数组赋任意值
    for (int i=0;i<input_layer_node_number;i++) {//给输入层到第一隐藏层权重数组赋值；
        for (int j=0;j<first_hiden_layer_node_number; j++) {
            par.ih_weight_array[i][j]=par.sample_from_gaussian(0,0.01);
        }
    }
    for (int i=0;i<first_hiden_layer_node_number;i++) {//给第一隐藏层到第二隐藏层权重数组赋值；
        for (int j=0;j<second_hiden_layer_node_number; j++) {
            par.hh_weight_array[i][j]=par.sample_from_gaussian(0,0.01);
        }
    }
    for (int i=0;i<second_hiden_layer_node_number;i++) {//给第二隐藏层到输出层权重数组赋值；
        for (int j=0;j<output_layer_node_number; j++) {
            par.ho_weight_array[i][j]=par.sample_from_gaussian(0,0.01);
        }
    }
    for (int i=0;i<first_hiden_layer_node_number; i++) {
        par.first_hiden_layer_bias_array[i]=par.sample_from_gaussian(0,0.01);
    }
    for (int i=0;i<first_hiden_layer_node_number; i++) {
        par.first_hiden_layer_bias_array[i]=par.sample_from_gaussian(0,0.01);
    }
    for (int i=0;i<first_hiden_layer_node_number; i++) {
        par.first_hiden_layer_bias_array[i]=par.sample_from_gaussian(0,0.01);
    }
    feedforward(inp,par);
    feedback(nva,par);
  return 0;
}

