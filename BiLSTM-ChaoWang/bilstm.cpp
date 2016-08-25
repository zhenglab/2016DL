#include <iostream>
#include <utility>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <cfloat>
#include <vector>
#include <fstream>

//#include "util.h"

using namespace std;


typedef vector< vector<double> > D2array; //二维数组
typedef vector<double> D1array;

/* bidrectional lstm
 *
 *
*/
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
};

double tanh(double x)
{
    return 2.0 * 1.0 / (1.0 + exp(-2.0 * x)) - 1.0;
};

double dsigmoid(float x)
{
    return (1.0 / (1.0 + exp(-x))) * (1 - 1.0 / (1.0 + exp(-x)));
};

double dtanh(float x)
{
    return 1 - pow((2.0 * 1.0 / (1.0 + exp(-2.0 * x)) - 1.0),2);
};

/***************************************************************************************************/

//fill weights

double gaussian(float mean, float variance)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 )
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        }
        while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);

    X = X * variance + mean;
    phase = 1 - phase;

    return X;
};


/* init weights
void init_weights()
{
    //method of guassian

    for(int d = 0; d < H_; ++d)
    {


        //cell weights
        c_weight_i[d] = gaussian(0, 0.1);
        c_weight_f[d] = gaussian(0, 0.1);
        c_weight_o[d] = gaussian(0, 0.1);

        for(int i = 0; i < I_; ++i)
        {
            weight_i[d][i] = gaussian(0, 0.1);
            weight_f[d][i] = gaussian(0, 0.1);
            weight_g[d][i] = gaussian(0, 0.1);
            weight_o[d][i] = gaussian(0, 0.1);

        }

        for(int j = 0; j < H_; ++j)
        {
            h_weight_i[d][j] = gaussian(0, 0.1);
            h_weight_f[d][j] = gaussian(0, 0.1);
            h_weight_g[d][j] = gaussian(0, 0.1);
            h_weight_o[d][j] = gaussian(0, 0.1);
        }
    }


}
*/
//softmax
double softmax(double *x)
{

    double max = 0.0;
    double sum = 0.0;

    for(int i = 0; i<8; ++i) if(max < x[i]) max = x[i];
    for(int j = 0; j<8; ++j)
    {
        x[j] = exp(x[j] - max);
        sum += x[j];
    }

    for(int l = 0; l<8; ++l) x[l] /=sum;

}


//shuffle
void perfect_shuffle(int *a,int n)
{
    int t,i;
    if (n == 1)
    {
        t = a[1];
        a[1] = a[2];
        a[2] = t;
        return;
    }
    int n2 = n * 2, n3 = n / 2;
    if (n % 2 == 1)    //奇数的处理
    {
        t = a[n];
        for (i = n + 1; i <= n2; ++i)
        {
            a[i - 1] = a[i];
        }
        a[n2] = t;
        --n;
    }
    //到此n是偶数

    for (i = n3 + 1; i <= n; ++i)
    {
        t = a[i];
        a[i] = a[i + n3];
        a[i + n3] = t;
    }

    // [1.. n /2]
    perfect_shuffle(a, n3);
    perfect_shuffle(a + n, n3);
}






//bidirectional LSTM

int main()
{
    /*
        int input_units = 6;
        int hidden_units = 10;
        int output_units = 8;
        int n_samples = 8;
        int steps=300;
    */

    /*    double x[2400][6];
        float y[8] = {0,1,2,3,4,5,6};

        for(int m = 0; m < 2400; ++m)
        {
            for(int n = 0; n < 6; ++n)
            {
                x[m][n] = gaussian(1, 0.1);
            }
        }

    */

    int H_ = 100; // hidden units
    int I_ = 6; // input units
    int T_ = 300; // steps
    int N_ = 64; //samples
    int K_ = 8;   // output units

    float lr=0.0001;
    int batch_count = 1;
    int batch_size = 1;
    float best_accury = 0.0;  //save the best modle

    //forward layer
    double f_weight_i[H_][I_];   //num of cell block
    double f_weight_f[H_][I_];
    double f_weight_g[H_][I_];   //candidate state
    double f_weight_o[H_][I_];

    double f_h_weight_i[H_][H_];
    double f_h_weight_f[H_][H_];
    double f_h_weight_g[H_][H_];
    double f_h_weight_o[H_][H_];

    double f_c_weight_i[H_];
    double f_c_weight_f[H_];
    double f_c_weight_o[H_];

    double f_h_z_weight[K_][H_];  // hidden to output


    double f_bias_i[H_]; // forward bias
    double f_bias_f[H_];
    double f_bias_g[H_];
    double f_bias_o[H_];

    double f_c_t[T_][H_]; //forward
    double f_h_t[T_][H_];

    double f_net_i[T_][H_];
    double f_net_f[T_][H_];
    double f_net_g[T_][H_];
    double f_net_o[T_][H_];



    double f_gate_i[T_][H_];
    double f_gate_f[T_][H_];
    double f_state_g[T_][H_];
    double f_gate_o[T_][H_];

    double f_h_prev[H_];
    double f_c_prev[H_];


    double f_c_state_diff[T_][H_];
    double f_g_gate_diff[T_][H_];
    double f_f_gate_diff[T_][H_];
    double f_i_gate_diff[T_][H_];
    double f_o_gate_diff[T_][H_];

    double f_h_gate_diff[T_][H_]; //cell output diff
    double f_h_i_gate_diff[T_][H_]; // to compute h_gate_diff
    double f_h_f_gate_diff[T_][H_];
    double f_h_g_gate_diff[T_][H_];
    double f_h_o_gate_diff[T_][H_];

    double f_o_h_diff[T_][K_];



    //backward layer
    double b_weight_i[H_][I_];   //input to i f g o
    double b_weight_f[H_][I_];
    double b_weight_g[H_][I_];   //candidate state
    double b_weight_o[H_][I_];

    double b_h_weight_i[H_][H_]; // hidden to hidden
    double b_h_weight_f[H_][H_];
    double b_h_weight_g[H_][H_];
    double b_h_weight_o[H_][H_];

    //cell weughts
    double b_c_weight_i[H_];
    double b_c_weight_f[H_];
    double b_c_weight_o[H_];

    double b_h_z_weight[K_][H_];  // hidden to output

    double b_bias_i[H_]; // backward layer bias
    double b_bias_f[H_];
    double b_bias_g[H_];
    double b_bias_o[H_];

    double b_c_t[T_][H_]; //backward layer
    double b_h_t[T_][H_];


    double b_net_i[T_][H_];
    double b_net_f[T_][H_];
    double b_net_g[T_][H_];
    double b_net_o[T_][H_];



    double b_gate_i[T_][H_];
    double b_gate_f[T_][H_];
    double b_state_g[T_][H_];
    double b_gate_o[T_][H_];

    double b_h_prev[H_];
    double b_c_prev[H_];


    double b_c_state_diff[T_][H_];
    double b_g_gate_diff[T_][H_];
    double b_f_gate_diff[T_][H_];
    double b_i_gate_diff[T_][H_];
    double b_o_gate_diff[T_][H_];

    double b_h_gate_diff[T_][H_]; //cell output diff
    double b_h_i_gate_diff[T_][H_]; // to compute h_gate_diff
    double b_h_f_gate_diff[T_][H_];
    double b_h_g_gate_diff[T_][H_];
    double b_h_o_gate_diff[T_][H_];

    double b_o_h_diff[T_][K_];





    //weights saved for test when test accuary > best_accuary;
    double save_weight_i[H_][I_];   //num of cell block
    double save_weight_f[H_][I_];
    double save_weight_g[H_][I_];   //candidate state
    double save_weight_o[H_][I_];

    double save_h_weight_i[H_][H_]; //hidden to hidden weights
    double save_h_weight_f[H_][H_];
    double save_h_weight_g[H_][H_];
    double save_h_weight_o[H_][H_];


    double save_c_weight_i[H_]; //cell block weights
    double save_c_weight_f[H_];
    double save_c_weight_o[H_];

    double save_bias_i[H_]; // bias saved
    double save_bias_f[H_];
    double save_bias_g[H_];
    double save_bias_o[H_];
    double save_bias_z[K_];




    //output layer
    double bias_z[K_]; //bias of hidden to output
    double z_o[T_][K_]; //output layer

    //for test
    double z_o_test[T_][K_];
    double z_[K_];
    double z_test[K_];
    double *ip = z_;
    double *test_ip = z_test;


    double f_error[T_][H_];
    double b_error[T_][H_];
    double gradient_O_h[T_][K_];


    double softmax_o[T_][K_];
    double softmax_loss[T_][K_];
    double loss;

    //double elapsed_time; //compute time of a iteration

    double sum = 0;

    // data

    int row = 19200;
    int col = 6;

    int count = 0;




    int y[64];
    int y_test[64];
    int id[64];

    //double x[row][col];
    //float out[row][col];
    int label = 0;

    for(int i = 0; i<64; i++)
    {

        y[i] = label;
        id[i] = i;
        y_test[i] = label;
        if((i+1)%8 == 0) label++;
        //cout << id[i]  << endl;

    }

    ifstream input_train("nn_train.txt");//打开输入文件 载入训练数据
    ifstream input_test("nn_test.txt"); //载入测试数据
    cout << "error2" << endl;
    //ofstream output("E:\\c++\\C++ code\\item_basedCF\\mytext.txt");     //打开要写入的文件，如果该文件不存在，则自动
    D2array x(row, D1array (col, 0));                                   //声明一个二维数组，将读入的数据写入该数组
    D2array x_test(row, D1array (col, 0));

    if (!input_train.is_open())     //如果文件打开失败
    {
        cout << "File is not existing!" << endl;
        exit(1);
    }
    if (!input_test.is_open())     //如果文件打开失败
    {
        cout << "File is not existing!" << endl;
        exit(1);
    }

    for (int i = 0; i < row; ++i)
    {


        for (int j = 0; j < col; j++)
        {
            input_train >> x[i][j];  //从输入流对象input读取字符到out
            input_test >> x_test[i][j];
            //cout << x_test[i][j] << " ";
            //x[i][j] = out[i][j];
            count++;
        }
        //cout << " " << endl;
        //output << endl;
    }
    //cout << "count is 19200" << endl;
    //cout << count << endl;

    input_train.close();
    input_test.close();


    // init weights
    for(int d = 0; d < H_; ++d)
    {

        //cell weights
        f_c_weight_i[d] = gaussian(0, 0.1);
        f_c_weight_f[d] = gaussian(0, 0.1);
        f_c_weight_o[d] = gaussian(0, 0.1);

        b_c_weight_i[d] = gaussian(0, 0.1);
        b_c_weight_f[d] = gaussian(0, 0.1);
        b_c_weight_o[d] = gaussian(0, 0.1);


        f_bias_i[d] = 0;
        f_bias_f[d] = 0;
        f_bias_g[d] = 0;
        f_bias_o[d] = 0;

        b_bias_i[d] = 0;
        b_bias_f[d] = 0;
        b_bias_g[d] = 0;
        b_bias_o[d] = 0;

        for(int i = 0; i < I_; ++i)
        {
            f_weight_i[d][i] = gaussian(0, 0.1);
            f_weight_f[d][i] = gaussian(0, 0.1);
            f_weight_g[d][i] = gaussian(0, 0.1);
            f_weight_o[d][i] = gaussian(0, 0.1);

            b_weight_i[d][i] = gaussian(0, 0.1);
            b_weight_f[d][i] = gaussian(0, 0.1);
            b_weight_g[d][i] = gaussian(0, 0.1);
            b_weight_o[d][i] = gaussian(0, 0.1);


        }

        for(int j = 0; j < H_; ++j)
        {
            f_h_weight_i[d][j] = gaussian(0, 0.1);
            f_h_weight_f[d][j] = gaussian(0, 0.1);
            f_h_weight_g[d][j] = gaussian(0, 0.1);
            f_h_weight_o[d][j] = gaussian(0, 0.1);

            b_h_weight_i[d][j] = gaussian(0, 0.1);
            b_h_weight_f[d][j] = gaussian(0, 0.1);
            b_h_weight_g[d][j] = gaussian(0, 0.1);
            b_h_weight_o[d][j] =gaussian(0, 0.1);

        }
        //previous state
        for(int k = 0; k < K_; ++k)
        {
            f_h_z_weight[k][d] = gaussian(0, 0.1);
            b_h_z_weight[k][d] = gaussian(0, 0.1);

        }
    }

// initialize h_prev[]

    for(int h = 0; h < H_; ++h)
    {
        f_h_prev[h] = 0;
        f_c_prev[h] = 0;
        b_h_prev[h] = 0;
        b_c_prev[h] = 0;


    }




    int max_epoch = 200;

// or
    for(int epoch = 0; epoch < max_epoch; ++epoch)
    {

        cout << epoch << " epochs" << endl;
        //elapsed_time = omp_get_wtime();
        perfect_shuffle(id, 31);
        perfect_shuffle(id, 31);
        perfect_shuffle(y, 31);
        perfect_shuffle(y, 31);

        for(int n = 0; n < N_; ++n)
        {

            //cout << n << " iteration" << endl;

            // forwardpass of forward hidden layer

            for(int t = 0; t < T_; ++t)  // T_ step
            {
                //previous state


                for(int d = 0; d < H_; ++d)  // H_：hidden units
                {
                    // in, forget, g_sate, output
                    // compute like MLP
                    //cout << d << "hidden unit" << (n * T_ + t) << "x_input" << endl;

                    for(int i = 0; i < I_; ++i)
                    {
                        f_net_i[t][d] += f_weight_i[d][i] * x[id[n] * T_ + t][i] ;
                        f_net_f[t][d] += f_weight_f[d][i] * x[id[n] * T_ + t][i] ;
                        f_net_g[t][d] += f_weight_g[d][i] * x[id[n] * T_ + t][i] ;
                        f_net_o[t][d] += f_weight_o[d][i] * x[id[n] * T_ + t][i] ;

                        //cout << id[n] << " id and label " << y[n] << endl;
                        // test this block of code, x_input and bias are ok, weight_update is the problem
                    }

                    // !!! input is ok

                    // sum the h(t-1) state;  prev output的处理上应该存在问题

                    for(int j = 0; j < H_; ++j)
                    {

                        f_net_i[t][d] += f_h_weight_i[d][j] * f_h_prev[j];
                        f_net_f[t][d] += f_h_weight_f[d][j] * f_h_prev[j];
                        f_net_g[t][d] += f_h_weight_g[d][j] * f_h_prev[j];
                        f_net_o[t][d] += f_h_weight_o[d][j] * f_h_prev[j];

                        //std::cout << "h_prev" << h_prev[j] << "h_t" << h_t[t][j] << endl;

                    }


                    //sum c(t-1)
                    f_net_i[t][d] += f_c_weight_i[d] * f_c_prev[d];
                    f_net_f[t][d] += f_c_weight_f[d] * f_c_prev[d];

                    //compute input gate, forget gate, candiate cell state, output gate
                    f_gate_i[t][d] = sigmoid(f_net_i[t][d] + f_bias_i[d]);//input gate
                    f_gate_f[t][d] = sigmoid(f_net_f[t][d] + f_bias_f[d]);//forget gate
                    f_state_g[t][d] = tanh(f_net_g[t][d] + f_bias_g[d]);

                    //cell output
                    f_c_t[t][d] = f_gate_f[t][d] * f_c_prev[d] + f_gate_i[t][d] * f_state_g[t][d];

                    f_net_o[t][d] += f_c_weight_o[d] * f_c_t[t][d];

                    // output gate  多了一项c_t[t][d]
                    // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                    f_gate_o[t][d] = sigmoid(f_net_o[t][d] + f_bias_o[d]);//output gate

                    //cell netoutput one time step
                    f_h_t[t][d] = f_gate_o[t][d] * tanh(f_c_t[t][d]); //cell output

                    //cout << "h_t out" << h_t[t][d] << endl;

                }

                //update prev h&c
                for(int d=0; d<H_; ++d)
                {
                    f_h_prev[d] = f_h_t[t][d];
                    f_c_prev[d] = f_c_t[t][d];

                }

            }

            //forward pass of backward hidden layer
            for(int t = T_-1; t >= 0; --t)  // T_ step
            {
                //previous state

                for(int d = 0; d < H_; ++d)  // H_：hidden units
                {
                    // in, forget, g_sate, output
                    // compute like MLP
                    //cout << d << "hidden unit" << (n * T_ + t) << "x_input" << endl;

                    for(int i = 0; i < I_; ++i)
                    {
                        b_net_i[t][d] += b_weight_i[d][i] * x[id[n] * T_ + t][i] ;
                        b_net_f[t][d] += b_weight_f[d][i] * x[id[n] * T_ + t][i] ;
                        b_net_g[t][d] += b_weight_g[d][i] * x[id[n] * T_ + t][i] ;
                        b_net_o[t][d] += b_weight_o[d][i] * x[id[n] * T_ + t][i] ;

                        //cout << id[n] << " id and label " << y[n] << endl;
                        // test this block of code, x_input and bias are ok, weight_update is the problem
                    }

                    // !!! input is ok

                    // sum the h(t-1) state;  prev output的处理上应该存在问题

                    for(int j = 0; j < H_; ++j)
                    {

                        b_net_i[t][d] += b_h_weight_i[d][j] * b_h_prev[j];
                        b_net_f[t][d] += b_h_weight_f[d][j] * b_h_prev[j];
                        b_net_g[t][d] += b_h_weight_g[d][j] * b_h_prev[j];
                        b_net_o[t][d] += b_h_weight_o[d][j] * b_h_prev[j];

                        //std::cout << "h_prev" << h_prev[j] << "h_t" << h_t[t][j] << endl;

                    }


                    //sum c(t-1)
                    b_net_i[t][d] += b_c_weight_i[d] * b_c_prev[d];
                    b_net_f[t][d] += b_c_weight_f[d] * b_c_prev[d];

                    //compute input gate, forget gate, candiate cell state, output gate
                    b_gate_i[t][d] = sigmoid(b_net_i[t][d] + b_bias_i[d]);//input gate
                    b_gate_f[t][d] = sigmoid(b_net_f[t][d] + b_bias_f[d]);//forget gate
                    b_state_g[t][d] = tanh(b_net_g[t][d] + b_bias_g[d]);

                    //cell output
                    b_c_t[t][d] = b_gate_f[t][d] * b_c_prev[d] + b_gate_i[t][d] * b_state_g[t][d];

                    b_net_o[t][d] += b_c_weight_o[d] * b_c_t[t][d];

                    // output gate  多了一项c_t[t][d]
                    // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                    b_gate_o[t][d] = sigmoid(b_net_o[t][d] + b_bias_o[d]);//output gate

                    //cell netoutput one time step
                    b_h_t[t][d] = b_gate_o[t][d] * tanh(b_c_t[t][d]); //cell output

                    //cout << "h_t out" << h_t[t][d] << endl;

                }

                //update prev h&c
                for(int d=0; d<H_; ++d)
                {
                    b_h_prev[d] = b_h_t[t][d];
                    b_c_prev[d] = b_c_t[t][d];

                }



            }


            // forward is ok  未出现 core dumped
            // compute
            // output layer

            for(int t=0; t<T_; ++t)
            {

                //std::cout << t <<"step" << std::endl;

                for(int f = 0; f < K_; ++f)

                {
                    for(int h = 0; h < H_; ++h)
                    {
                        z_o[t][f] += f_h_z_weight[f][h] * f_h_t[t][h] + b_h_z_weight[f][h] * b_h_t[t][h];

                    }

                    z_o[t][f] = z_o[t][f] + bias_z[f];
                    z_[f] = z_o[t][f];
                    //cout << f << " bias_z:" << bias_z[f] << "z_o" << z_[f] << std::endl;
                    //at time step computesoftmax
                }


                softmax(z_);
                for(int f = 0; f < K_; ++f)
                {

                    softmax_o[t][f] = *ip;
                    sum += softmax_o[t][f];
                    //std::cout << "the probability of" << f << "output units:" << " "
                    //   << "lable"  <<id[n]/8 <<  " "
                    //  << softmax_o[t][f] << std::endl;
                    ip++;
                    softmax_loss[t][f] = -log(softmax_o[t][f]);

                }
                //std::cout << "sum" << std::endl;
                //std::cout << sum << std::endl;
                sum = 0;

                for(int k = 0; k < K_; ++k)
                {
                    loss += softmax_loss[t][k];

                }
                ip = z_;
                //num++;

            }
            //cout << "loss" << endl;
            //cout << loss << endl;


          
            //  before backward is ok 程序可以执行到此处



            //forward layer Backward pass
            for( int t = T_-1; t >= 0; --t)
            {
                //compute error, T step , cell outputs

                for(int h = 0; h < H_; ++h)
                {
                    for(int k =0; k<K_; ++k)
                    {
                        if(y[id[n]] == k)
                        {
                            f_error[t][h] += f_h_z_weight[k][h]*(softmax_o[t][k] - 1);
                            //cout << "compute" << endl;
                        }
                        else
                        {
                            f_error[t][h] += f_h_z_weight[k][h]  * softmax_o[t][k];
                        }

                    }

                }

                for(int k = 0; k<K_; ++k)
                {
                    if(y[id[n]] == k)
                    {
                        gradient_O_h[t][k] = softmax_o[t][k] - 1;
                    }
                    else
                    {
                        gradient_O_h[t][k] = softmax_o[t][k];
                    }

                }



                for(int d = 0; d < H_; ++d)
                {

                    //float error = h_t[t][d] - y[t][d];
                    //float dEdo = tanh(c_t[t][H_-d]);




                    //error += (z_o[t][k] - y[k]) * h_z_weight[K_][H_];

                    //float dEdc = gate_t[2*H_ + d] * (1 - pow(tanh(c_t[t][H_-d])),2);

                    //E[H_-d] = 0.5 * pow(error, 2);
                    //follow Graves

                    if( t==T_)
                    {
                        // cell state
                        // cell output is error

                        //cell outputs
                        f_h_i_gate_diff[t][d] = 0;
                        f_h_f_gate_diff[t][d] = 0;
                        f_h_g_gate_diff[t][d] = 0;
                        f_h_o_gate_diff[t][d] = 0;

                        f_h_gate_diff[t][d] = f_error[t][d] + f_h_i_gate_diff[t][d] + f_h_f_gate_diff[t][d] + f_h_g_gate_diff[t][d] + f_h_o_gate_diff[t][d];
                        //output gates
                        f_o_gate_diff[t][d] = dsigmoid(f_net_o[t][d]) * f_h_gate_diff[t][d] * tanh(f_gate_o[t][d]);
                        //cell state
                        f_c_state_diff[t][d] = f_gate_o[t][d] * dtanh(f_c_t[t][d]) * f_h_gate_diff[t][d] +  f_c_weight_o[d]*f_o_gate_diff[t][d] ;
                        //candidate cell state
                        f_g_gate_diff[t][d] = f_gate_i[t][d] * dtanh(f_state_g[t][d]) * f_c_state_diff[t][d];
                        //forget gate
                        f_f_gate_diff[t][d] = dsigmoid(f_net_f[t][d]) * f_c_t[t-1][d] * f_c_state_diff[t][d];  //公式没有错误。。。
                        //input gate
                        f_i_gate_diff[t][d] = dsigmoid(f_net_i[t][d]) * f_c_state_diff[t][d] * f_state_g[t][d]; // input 有一个错误，dsigmoid应为net_i

                        // 改进该错误之后，c_weight 爆炸没有那么
                        // 快了 32sample 为nan

                    }
                    else
                    {
                        // T-1 then diff
                        //cell outputs
                        for(int hi = 0; hi < H_; ++hi)
                        {
                            f_h_i_gate_diff[t][d] += f_error[t+1][d] * f_gate_o[t+1][d] * dtanh(f_c_t[t+1][d]) * f_state_g[t+1][d] * dsigmoid(f_net_i[t+1][d]) * f_h_weight_i[d][hi];
                            f_h_f_gate_diff[t][d] += f_error[t+1][d] * f_gate_o[t+1][d] * dtanh(f_c_t[t+1][d]) * f_c_t[t][d] * dsigmoid(f_net_f[t+1][d]) * f_h_weight_f[d][hi];
                            f_h_g_gate_diff[t][d] += f_error[t+1][d] * f_gate_o[t+1][d] * dtanh(f_c_t[t+1][d]) * f_gate_i[t+1][d] * dtanh(f_net_g[t+1][d]) * f_h_weight_g[d][hi];
                            f_h_o_gate_diff[t][d] += f_error[t+1][d] * tanh(f_c_t[t+1][d]) * dsigmoid(f_gate_o[t+1][d]) * f_h_weight_o[d][hi];
                        }
                        f_h_gate_diff[t][d] = f_error[t][d] + f_h_i_gate_diff[t][d] + f_h_f_gate_diff[t][d] + f_h_g_gate_diff[t][d] + f_h_o_gate_diff[t][d];
                        //output gate
                        f_o_gate_diff[t][d] = dsigmoid(f_net_o[t][d]) * f_h_gate_diff[t][d] * tanh(f_gate_o[t][d]);
                        //cell state
                        f_c_state_diff[t][d] = f_gate_o[t][d] * dtanh(f_c_t[t][d]) * f_error[t][d] + f_gate_f[t+1][d] * f_c_state_diff[t+1][d] + f_c_weight_i[d] * f_i_gate_diff[t+1][d] + f_c_weight_f[d] * f_f_gate_diff[t+1][d] + f_c_weight_o[d] * f_o_gate_diff[t][d];
                        //candidate cell state
                        f_g_gate_diff[t][d] = f_gate_i[t][d] * dtanh(f_state_g[t][d]) * f_c_state_diff[t][d];
                        //forget gate
                        f_f_gate_diff[t][d] = dsigmoid(f_net_f[t][d]) * f_c_t[t-1][d] * f_c_state_diff[t][d];
                        //input gate
                        f_i_gate_diff[t][d] = dsigmoid(f_net_i[t][d]) * f_c_state_diff[t][d] * f_state_g[t][d];




                    }


                }

            }



            //backward layer for backward pass

            for( int t = 0; t < T_; ++t)
            {
                //compute error, T step , cell outputs

                for(int h = 0; h < H_; ++h)
                {
                    for(int k =0; k<K_; ++k)
                    {
                        if(y[id[n]] == k)
                        {
                            b_error[t][h] += b_h_z_weight[k][h]*(softmax_o[t][k] - 1);
                            //cout << "compute" << endl;
                        }
                        else
                        {
                            b_error[t][h] +=  b_h_z_weight[k][h] * softmax_o[t][k];
                        }

                    }

                }

                for(int k = 0; k<K_; ++k)
                {
                    if(y[id[n]] == k)
                    {
                        gradient_O_h[t][k] = softmax_o[t][k] - 1;
                    }
                    else
                    {
                        gradient_O_h[t][k] = softmax_o[t][k];
                    }

                }



                for(int d = 0; d < H_; ++d)
                {

                    //float error = h_t[t][d] - y[t][d];
                    //float dEdo = tanh(c_t[t][H_-d]);




                    //error += (z_o[t][k] - y[k]) * h_z_weight[K_][H_];

                    //float dEdc = gate_t[2*H_ + d] * (1 - pow(tanh(c_t[t][H_-d])),2);

                    //E[H_-d] = 0.5 * pow(error, 2);
                    //follow Graves

                    if( t==T_)
                    {
                        // cell state
                        // cell output is error

                        //cell outputs
                        b_h_i_gate_diff[t][d] = 0;
                        b_h_f_gate_diff[t][d] = 0;
                        b_h_g_gate_diff[t][d] = 0;
                        b_h_o_gate_diff[t][d] = 0;

                        b_h_gate_diff[t][d] = b_error[t][d] + b_h_i_gate_diff[t][d] + b_h_f_gate_diff[t][d] + b_h_g_gate_diff[t][d] + b_h_o_gate_diff[t][d];
                        //output gates
                        b_o_gate_diff[t][d] = dsigmoid(b_net_o[t][d]) * b_h_gate_diff[t][d] * tanh(b_gate_o[t][d]);
                        //cell state
                        b_c_state_diff[t][d] = b_gate_o[t][d] * dtanh(b_c_t[t][d]) * b_h_gate_diff[t][d] +  b_c_weight_o[d]*b_o_gate_diff[t][d] ;
                        //candidate cell state
                        b_g_gate_diff[t][d] = b_gate_i[t][d] * dtanh(b_state_g[t][d]) * b_c_state_diff[t][d];
                        //forget gate
                        b_f_gate_diff[t][d] = dsigmoid(b_net_f[t][d]) * b_c_t[t-1][d] * b_c_state_diff[t][d];  //公式没有错误。。。
                        //input gate
                        b_i_gate_diff[t][d] = dsigmoid(b_net_i[t][d]) * b_c_state_diff[t][d] * b_state_g[t][d]; // input 有一个错误，dsigmoid应为net_i

                        // 改进该错误之后，c_weight 爆炸没有那么
                        // 快了 32sample 为nan

                    }
                    else
                    {
                        // T-1 then diff
                        //cell outputs
                        for(int hi = 0; hi < H_; ++hi)
                        {
                            b_h_i_gate_diff[t][d] += b_error[t+1][d] * b_gate_o[t+1][d] * dtanh(b_c_t[t+1][d]) * b_state_g[t+1][d] * dsigmoid(b_net_i[t+1][d]) * b_h_weight_i[d][hi];
                            b_h_f_gate_diff[t][d] += b_error[t+1][d] * b_gate_o[t+1][d] * dtanh(b_c_t[t+1][d]) * b_c_t[t][d] * dsigmoid(b_net_f[t+1][d]) * b_h_weight_f[d][hi];
                            b_h_g_gate_diff[t][d] += b_error[t+1][d] * b_gate_o[t+1][d] * dtanh(b_c_t[t+1][d]) * b_gate_i[t+1][d] * dtanh(b_net_g[t+1][d]) * b_h_weight_g[d][hi];
                            b_h_o_gate_diff[t][d] += b_error[t+1][d] * tanh(b_c_t[t+1][d]) * dsigmoid(b_gate_o[t+1][d]) * b_h_weight_o[d][hi];
                        }
                        b_h_gate_diff[t][d] = b_error[t][d] + b_h_i_gate_diff[t][d] + b_h_f_gate_diff[t][d] + b_h_g_gate_diff[t][d] + b_h_o_gate_diff[t][d];
                        //output gate
                        b_o_gate_diff[t][d] = dsigmoid(b_net_o[t][d]) * b_h_gate_diff[t][d] * tanh(b_gate_o[t][d]);
                        //cell state
                        b_c_state_diff[t][d] = b_gate_o[t][d] * dtanh(b_c_t[t][d]) * b_error[t][d] + b_gate_f[t+1][d] * b_c_state_diff[t+1][d] + b_c_weight_i[d] * b_i_gate_diff[t+1][d] + b_c_weight_f[d] * b_f_gate_diff[t+1][d] + b_c_weight_o[d] * b_o_gate_diff[t][d];
                        //candidate cell state
                        b_g_gate_diff[t][d] = b_gate_i[t][d] * dtanh(b_state_g[t][d]) * b_c_state_diff[t][d];
                        //forget gate
                        b_f_gate_diff[t][d] = dsigmoid(b_net_f[t][d]) * b_c_t[t-1][d] * b_c_state_diff[t][d];
                        //input gate
                        b_i_gate_diff[t][d] = dsigmoid(b_net_i[t][d]) * b_c_state_diff[t][d] * b_state_g[t][d];




                    }


                }

            }
            //batch_count++;
            // before update is ok （core dumped problem）

            if(batch_count == batch_size) // version of batch = 1
            {
                // update weights and bias
                batch_count = 0;
                // cout << n << "sample:" << " " <<endl ;
                for(int t = T_ - 1; t >= 0; --t)
                {
                    // cout << t << "steps:" << " " <<endl ;

                    for(int d = 0; d < H_; ++d)
                    {

                        f_c_weight_i[d] = f_c_weight_i[d] - lr * f_i_gate_diff[t][d] * f_c_t[t-1][d];
                        f_c_weight_f[d] = f_c_weight_f[d] - lr * f_f_gate_diff[t][d] * f_c_t[t-1][d];
                        f_c_weight_o[d] = f_c_weight_o[d] - lr * f_g_gate_diff[t][d] * f_c_t[t][d];

                        b_c_weight_i[d] = b_c_weight_i[d] - lr * b_i_gate_diff[t][d] * b_c_t[t-1][d];
                        b_c_weight_f[d] = b_c_weight_f[d] - lr * b_f_gate_diff[t][d] * b_c_t[t-1][d];
                        b_c_weight_o[d] = b_c_weight_o[d] - lr * b_g_gate_diff[t][d] * b_c_t[t][d];

                        //cout << "c_weight_i:" << " " << c_weight_i[d] << " "
                        //       << "c_weight_f:" << " " << c_weight_f[d] << " "
                        //        << "c_weight_o:" << " " << c_weight_o[d] << endl;


                        for(int i = 0; i < I_; ++i)
                        {

                            f_weight_i[d][i] = f_weight_i[d][i] - lr * f_i_gate_diff[t][d] * x[n * T_ + t][i];
                            f_weight_f[d][i] = f_weight_f[d][i] - lr * f_f_gate_diff[t][d] * x[n * T_ + t][i];
                            f_weight_g[d][i] = f_weight_g[d][i] - lr * f_g_gate_diff[t][d] * x[n * T_ + t][i];
                            f_weight_o[d][i] = f_weight_o[d][i] - lr * f_o_gate_diff[t][d] * x[n * T_ + t][i];


                            b_weight_i[d][i] = b_weight_i[d][i] - lr * b_i_gate_diff[t][d] * x[n * T_ + t][i];
                            b_weight_f[d][i] = b_weight_f[d][i] - lr * b_f_gate_diff[t][d] * x[n * T_ + t][i];
                            b_weight_g[d][i] = b_weight_g[d][i] - lr * b_g_gate_diff[t][d] * x[n * T_ + t][i];
                            b_weight_o[d][i] = b_weight_o[d][i] - lr * b_o_gate_diff[t][d] * x[n * T_ + t][i];


                            //cout << h_z_weight[k][d] << endl;
                        }
                        //update weights of hidden to hidden

                        if(t < T_ - 1)
                        {

                            for(int h = 0; h<H_; ++h)
                            {
                                f_h_weight_i[d][h] = f_h_weight_i[d][h] - lr * f_h_i_gate_diff[t+1][h];
                                f_h_weight_f[d][h] = f_h_weight_f[d][h] - lr * f_h_f_gate_diff[t+1][h];
                                f_h_weight_g[d][h] = f_h_weight_g[d][h] - lr * f_h_g_gate_diff[t+1][h];
                                f_h_weight_o[d][h] = f_h_weight_o[d][h] - lr * f_h_o_gate_diff[t+1][h];

                                b_h_weight_i[d][h] = b_h_weight_i[d][h] - lr * b_h_i_gate_diff[t+1][h];
                                b_h_weight_f[d][h] = b_h_weight_f[d][h] - lr * b_h_f_gate_diff[t+1][h];
                                b_h_weight_g[d][h] = b_h_weight_g[d][h] - lr * b_h_g_gate_diff[t+1][h];
                                b_h_weight_o[d][h] = b_h_weight_o[d][h] - lr * b_h_o_gate_diff[t+1][h];

                            }
                        }

                        //hidden to output weights update error
                        for(int k = 0; k<K_; ++k)
                        {
                            f_h_z_weight[k][d] = f_h_z_weight[k][d] - lr*gradient_O_h[t][k] * f_h_t[t][d];
                            b_h_z_weight[k][d] = f_h_z_weight[k][d] - lr*gradient_O_h[t][k] * b_h_t[t][d];
                            //cout << "h_z_weight:" << "" << h_z_weight[k][d] << endl;
                            bias_z[k] = bias_z[k] - lr * gradient_O_h[t][k] ;
                        }

                        //bias of hidden to output
                        f_bias_i[d] = f_bias_i[d] - lr * f_i_gate_diff[t][d];
                        f_bias_f[d] = f_bias_f[d] - lr * f_f_gate_diff[t][d];
                        f_bias_g[d] = f_bias_g[d] - lr * f_g_gate_diff[t][d];
                        f_bias_o[d] = f_bias_o[d] - lr * f_o_gate_diff[t][d];

                        b_bias_i[d] = b_bias_i[d] - lr * b_i_gate_diff[t][d];
                        b_bias_f[d] = b_bias_f[d] - lr * b_f_gate_diff[t][d];
                        b_bias_g[d] = b_bias_g[d] - lr * b_g_gate_diff[t][d];
                        b_bias_o[d] = b_bias_o[d] - lr * b_o_gate_diff[t][d];

                        //test predict accuary every batch

                    }
                }
             }
                cout << "after update is ok"  << endl;
                // 将每个sample的值清零（for test）


                for(int t_clear = 0; t_clear < T_; ++t_clear)
                {
                    //cout << "clear"<< endl;

                    for(int d = 0; d < H_; ++d)
                    {

                        f_c_t[t_clear][d] = 0;
                        f_h_t[t_clear][d] = 0;
                        f_net_i[t_clear][d] = 0;
                        f_net_f[t_clear][d] = 0;
                        f_net_g[t_clear][d] = 0;
                        f_net_o[t_clear][d] = 0;
                        f_gate_i[t_clear][d] = 0;
                        f_gate_f[t_clear][d] = 0;
                        f_state_g[t_clear][d] = 0;
                        f_gate_o[t_clear][d] = 0;

                        // 梯度可以 在batch时累加求均值
                        f_c_state_diff[t_clear][d] = 0;
                        f_g_gate_diff[t_clear][d] = 0;
                        f_f_gate_diff[t_clear][d] = 0;
                        f_i_gate_diff[t_clear][d] = 0;
                        f_o_gate_diff[t_clear][d] = 0;
                        f_h_gate_diff[t_clear][d] = 0; //cell output diff
                        f_h_i_gate_diff[t_clear][d] = 0;
                        f_h_f_gate_diff[t_clear][d] = 0;
                        f_h_g_gate_diff[t_clear][d] = 0;
                        f_h_o_gate_diff[t_clear][d] = 0;
                        f_error[t_clear][d] = 0;

                        f_h_prev[d] = 0;
                        f_c_prev[d] = 0;

                        b_c_t[t_clear][d] = 0;
                        b_h_t[t_clear][d] = 0;
                        b_net_i[t_clear][d] = 0;
                        b_net_f[t_clear][d] = 0;
                        b_net_g[t_clear][d] = 0;
                        b_net_o[t_clear][d] = 0;
                        b_gate_i[t_clear][d] = 0;
                        b_gate_f[t_clear][d] = 0;
                        b_state_g[t_clear][d] = 0;
                        b_gate_o[t_clear][d] = 0;

                        // 梯度可以 在batch时累加求均值
                        b_c_state_diff[t_clear][d] = 0;
                        b_g_gate_diff[t_clear][d] = 0;
                        b_f_gate_diff[t_clear][d] = 0;
                        b_i_gate_diff[t_clear][d] = 0;
                        b_o_gate_diff[t_clear][d] = 0;
                        b_h_gate_diff[t_clear][d] = 0; //cell output diff
                        b_h_i_gate_diff[t_clear][d] = 0;
                        b_h_f_gate_diff[t_clear][d] = 0;
                        b_h_g_gate_diff[t_clear][d] = 0;
                        b_h_o_gate_diff[t_clear][d] = 0;
                        b_error[t_clear][d] = 0;

                        b_h_prev[d] = 0;
                        b_c_prev[d] = 0;

                    }

                    for(int k = 0; k < K_; ++k)
                    {
                        z_o[t_clear][k] = 0 ;
                        gradient_O_h[t_clear][k] = 0;
                        //o_h_diff[t_clear][k] = 0;
                        softmax_o[t_clear][k] = 0;
                        softmax_loss[t_clear][k] = 0;

                        z_[k] = 0; //


                    }

                }



                for(int test_iter = 0 ; test_iter < 1; ++test_iter)
                {
                    cout << "test iter" << endl;
                    double test_accuary = 0.0;
                    // forwardpass of forward hidden layer
                    for(int sample = 0; sample < N_; ++sample)
                    {
                        for(int t = 0; t < T_; ++t)  // T_ step
                        {
                            //previous state


                            for(int d = 0; d < H_; ++d)  // H_：hidden units
                            {
                                // in, forget, g_sate, output
                                // compute like MLP
                                //cout << d << "hidden unit" << (n * T_ + t) << "x_input" << endl;

                                for(int i = 0; i < I_; ++i)
                                {

                                    f_net_i[t][d] += f_weight_i[d][i] * x[id[sample] * T_ + t][i];
                                    f_net_f[t][d] += f_weight_f[d][i] * x[id[sample] * T_ + t][i];
                                    f_net_g[t][d] += f_weight_g[d][i] * x[id[sample] * T_ + t][i];
                                    f_net_o[t][d] += f_weight_o[d][i] * x[id[sample] * T_ + t][i];
                                    //cout << id[n] << " id and label " << y[n] << endl;
                                    // test this block of code, x_input and bias are ok, weight_update is the problem
                                }

                                // !!! input is ok

                                // sum the h(t-1) state;  prev output的处理上应该存在问题

                                for(int j = 0; j < H_; ++j)
                                {

                                    f_net_i[t][d] += f_h_weight_i[d][j] * f_h_prev[j];
                                    f_net_f[t][d] += f_h_weight_f[d][j] * f_h_prev[j];
                                    f_net_g[t][d] += f_h_weight_g[d][j] * f_h_prev[j];
                                    f_net_o[t][d] += f_h_weight_o[d][j] * f_h_prev[j];

                                    //std::cout << "h_prev" << h_prev[j] << "h_t" << h_t[t][j] << endl;

                                }


                                //sum c(t-1)
                                f_net_i[t][d] += f_c_weight_i[d] * f_c_prev[d];
                                f_net_f[t][d] += f_c_weight_f[d] * f_c_prev[d];

                                //compute input gate, forget gate, candiate cell state, output gate
                                f_gate_i[t][d] = sigmoid(f_net_i[t][d] + f_bias_i[d]);//input gate
                                f_gate_f[t][d] = sigmoid(f_net_f[t][d] + f_bias_f[d]);//forget gate
                                f_state_g[t][d] = tanh(f_net_g[t][d] + f_bias_g[d]);

                                //cell output
                                f_c_t[t][d] = f_gate_f[t][d] * f_c_prev[d] + f_gate_i[t][d] * f_state_g[t][d];

                                f_net_o[t][d] += f_c_weight_o[d] * f_c_t[t][d];

                                // output gate  多了一项c_t[t][d]
                                // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                                f_gate_o[t][d] = sigmoid(f_net_o[t][d] + f_bias_o[d]);//output gate

                                //cell netoutput one time step
                                f_h_t[t][d] = f_gate_o[t][d] * tanh(f_c_t[t][d]); //cell output

                                //cout << "h_t out" << h_t[t][d] << endl;

                            }

                            //update prev h&c
                            for(int d=0; d<H_; ++d)
                            {
                                f_h_prev[d] = f_h_t[t][d];
                                f_c_prev[d] = f_c_t[t][d];

                            }

                        }

                        //forward pass of backward hidden layer
                        for(int t = T_-1; t >= 0; --t)  // T_ step
                        {
                            //previous state

                            for(int d = 0; d < H_; ++d)  // H_：hidden units
                            {
                                // in, forget, g_sate, output
                                // compute like MLP
                                //cout << d << "hidden unit" << (n * T_ + t) << "x_input" << endl;

                                for(int i = 0; i < I_; ++i)
                                {
                                    b_net_i[t][d] += b_weight_i[d][i] * x[id[n] * T_ + t][i] ;
                                    b_net_f[t][d] += b_weight_f[d][i] * x[id[n] * T_ + t][i] ;
                                    b_net_g[t][d] += b_weight_g[d][i] * x[id[n] * T_ + t][i] ;
                                    b_net_o[t][d] += b_weight_o[d][i] * x[id[n] * T_ + t][i] ;

                                    //cout << id[n] << " id and label " << y[n] << endl;
                                    // test this block of code, x_input and bias are ok, weight_update is the problem
                                }

                                // !!! input is ok

                                // sum the h(t-1) state;  prev output的处理上应该存在问题

                                for(int j = 0; j < H_; ++j)
                                {

                                    b_net_i[t][d] += b_h_weight_i[d][j] * b_h_prev[j];
                                    b_net_f[t][d] += b_h_weight_f[d][j] * b_h_prev[j];
                                    b_net_g[t][d] += b_h_weight_g[d][j] * b_h_prev[j];
                                    b_net_o[t][d] += b_h_weight_o[d][j] * b_h_prev[j];

                                    //std::cout << "h_prev" << h_prev[j] << "h_t" << h_t[t][j] << endl;

                                }


                                //sum c(t-1)
                                b_net_i[t][d] += b_c_weight_i[d] * b_c_prev[d];
                                b_net_f[t][d] += b_c_weight_f[d] * b_c_prev[d];

                                //compute input gate, forget gate, candiate cell state, output gate
                                b_gate_i[t][d] = sigmoid(b_net_i[t][d] + b_bias_i[d]);//input gate
                                b_gate_f[t][d] = sigmoid(b_net_f[t][d] + b_bias_f[d]);//forget gate
                                b_state_g[t][d] = tanh(b_net_g[t][d] + b_bias_g[d]);

                                //cell output
                                b_c_t[t][d] = b_gate_f[t][d] * b_c_prev[d] + b_gate_i[t][d] * b_state_g[t][d];

                                b_net_o[t][d] += b_c_weight_o[d] * b_c_t[t][d];

                                // output gate  多了一项c_t[t][d]
                                // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                                b_gate_o[t][d] = sigmoid(b_net_o[t][d] + b_bias_o[d]);//output gate

                                //cell netoutput one time step
                                b_h_t[t][d] = b_gate_o[t][d] * tanh(b_c_t[t][d]); //cell output

                                //cout << "h_t out" << h_t[t][d] << endl;

                            }

                            //update prev h&c
                            for(int d=0; d<H_; ++d)
                            {
                                b_h_prev[d] = b_h_t[t][d];
                                b_c_prev[d] = b_c_t[t][d];

                            }



                        }


                    // forward is ok  未出现 core dumped
                    // compute
                    // output layer

                    for(int t=0; t<T_; ++t)
                    {

                        //std::cout << t <<"step" << std::endl;

                        for(int f = 0; f < K_; ++f)

                        {
                            for(int h = 0; h < H_; ++h)
                            {
                                z_o[t][f] += f_h_z_weight[f][h] * f_h_t[t][h] + b_h_z_weight[f][h] * b_h_t[t][h];

                            }

                            z_o[t][f] = z_o[t][f] + bias_z[f];
                            z_[f] = z_o[t][f];
                            //cout << f << " bias_z:" << bias_z[f] << "z_o" << z_[f] << std::endl;
                            //at time step computesoftmax
                        }


                        softmax(z_);
                        for(int f = 0; f < K_; ++f)
                        {

                            softmax_o[t][f] = *ip;
                            sum += softmax_o[t][f];
                            //std::cout << "the probability of" << f << "output units:" << " "
                            //   << "lable"  <<id[n]/8 <<  " "
                            //  << softmax_o[t][f] << std::endl;
                            ip++;
                            softmax_loss[t][f] = -log(softmax_o[t][f]);

                        }
                        //std::cout << "sum" << std::endl;
                        //std::cout << sum << std::endl;
                        sum = 0;

                        for(int k = 0; k < K_; ++k)
                        {
                            loss += softmax_loss[t][k];

                        }
                        ip = z_;
                        //num++;
                        double p_out_max = 0;
                                                double p_out[T_][K_];
                                                int y_predict = 0;

                                                for(int i = 0; i<K_; ++i)
                                                {
                                                    for(int t = 0; t<T_; ++t)
                                                        p_out[t][i] = 0;
                                                }

                                                for(int frame = 0; frame < T_; ++frame)
                                                {

                                                    // compute every output units "T" step probility

                                                    for(int out = 0; out < K_; ++out)
                                                    {


                                                        p_out[frame][out] = softmax_o[frame][out];

                                                        //cout << "p_out" << p_out[frame][out] << endl;

                                                        if(p_out[frame][out] > p_out_max )
                                                        {
                                                            p_out_max = p_out[frame][out];

                                                            y_predict = out;
                                                            //cout << y_predict << endl;

                                                        }


                                                    } //compute the

                                                    if(y_predict == y_test[sample])
                                                        {
                                                            test_accuary += 1 / double(300 * N_ );
                                                            //cout << "accuary change ：" << test_accuary << endl;
                                                        }
                    }
                   }
                    cout << "test_accuary :" << test_accuary << endl;

                }

                //clear end

            }

            for(int t_clear = 0; t_clear < T_; ++t_clear)
            {
                //cout << "clear"<< endl;

                for(int d = 0; d < H_; ++d)
                {
                    f_c_t[t_clear][d] = 0;
                    f_h_t[t_clear][d] = 0;
                    f_net_i[t_clear][d] = 0;
                    f_net_f[t_clear][d] = 0;
                    f_net_g[t_clear][d] = 0;
                    f_net_o[t_clear][d] = 0;
                    f_gate_i[t_clear][d] = 0;
                    f_gate_f[t_clear][d] = 0;
                    f_state_g[t_clear][d] = 0;
                    f_gate_o[t_clear][d] = 0;

                    // 梯度可以 在batch时累加
                    f_c_state_diff[t_clear][d] = 0;
                    f_g_gate_diff[t_clear][d] = 0;
                    f_f_gate_diff[t_clear][d] = 0;
                    f_i_gate_diff[t_clear][d] = 0;
                    f_o_gate_diff[t_clear][d] = 0;
                    f_h_gate_diff[t_clear][d] = 0; //cell output diff
                    f_h_i_gate_diff[t_clear][d] = 0;
                    f_h_f_gate_diff[t_clear][d] = 0;
                    f_h_g_gate_diff[t_clear][d] = 0;
                    f_h_o_gate_diff[t_clear][d] = 0;
                    f_error[t_clear][d] = 0;

                    f_h_prev[d] = 0;
                    f_c_prev[d] = 0;



                    b_c_t[t_clear][d] = 0;
                    b_h_t[t_clear][d] = 0;
                    b_net_i[t_clear][d] = 0;
                    b_net_f[t_clear][d] = 0;
                    b_net_g[t_clear][d] = 0;
                    b_net_o[t_clear][d] = 0;
                    b_gate_i[t_clear][d] = 0;
                    b_gate_f[t_clear][d] = 0;
                    b_state_g[t_clear][d] = 0;
                    b_gate_o[t_clear][d] = 0;

                    // 梯度可以 在batch时累加
                    b_c_state_diff[t_clear][d] = 0;
                    b_g_gate_diff[t_clear][d] = 0;
                    b_f_gate_diff[t_clear][d] = 0;
                    b_i_gate_diff[t_clear][d] = 0;
                    b_o_gate_diff[t_clear][d] = 0;
                    b_h_gate_diff[t_clear][d] = 0; //cell output diff
                    b_h_i_gate_diff[t_clear][d] = 0;
                    b_h_f_gate_diff[t_clear][d] = 0;
                    b_h_g_gate_diff[t_clear][d] = 0;
                    b_h_o_gate_diff[t_clear][d] = 0;
                    b_error[t_clear][d] = 0;

                    b_h_prev[d] = 0;
                    b_c_prev[d] = 0;
                }

                for(int k = 0; k < K_; ++k)
                {
                    z_o[t_clear][k] = 0 ;
                    gradient_O_h[t_clear][k] = 0;
                    softmax_o[t_clear][k] = 0;
                    softmax_loss[t_clear][k] = 0;
                    z_[k] = 0; //

                }

            }




    }

}
    return 0;

}



