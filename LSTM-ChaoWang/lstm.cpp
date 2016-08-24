#include <iostream>
#include <utility>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <cfloat>
#include <vector>
#include <fstream>

using namespace std;

typedef vector< vector<double> > D2array; //二维数组
typedef vector<double> D1array;

/* one layer lstm
 *
 *
*/

//define activation function
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

//softmax
double softmax(double *x)
{

    double max = 0.0;
    double sum = 0.0;

    //for(int i = 0; i<8; ++i) if(max < x[i]) max = x[i];
    for(int j = 0; j<8; ++j)
    {
        x[j] = exp(x[j]);
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



//forward pass right way to store data...
int main()
{
    /*
        int input_units = 6;
        int hidden_units = 10;
        int output_units = 8;
        int n_samples = 8;
        int steps=300;
    */

    int H_ = 200; // hidden units
    int I_ = 6; // input units
    int T_ = 300; // steps
    int N_ = 64; //samples
    int K_ = 8;   // output units

    float lr = 0.00064;
    int batch_count =0;
    double batch_size = 64.0;
    float best_accury = 0.0;  //save the best modle

    float weight_i[H_][I_];   //num of cell block
    float weight_f[H_][I_];
    float weight_g[H_][I_];   //candidate state
    float weight_o[H_][I_];

    float h_weight_i[H_][H_];
    float h_weight_f[H_][H_];
    float h_weight_g[H_][H_];
    float h_weight_o[H_][H_];

    //cell weughts
    float c_weight_i[H_];
    float c_weight_f[H_];
    float c_weight_o[H_];

    //weights saved for test when test accuary > best_accuary;


    //float softmax_o[N_];

    float h_z_weight[K_][H_];  // hidden to output

    // weights of
    // float weight_h[];  //num - 1

    float bias_i[H_]; //
    float bias_f[H_];
    float bias_g[H_];
    float bias_o[H_];
    float bias_z[K_]; //bias of hidden to output


    float c_t[T_][H_];
    float h_t[T_][H_];
    float h_sum[H_];
    float c_sum[H_];
    float z_o[T_][K_]; //output layer
    float z_o_test[T_][K_];
    double z_[K_];
    double z_test[K_];
    double *ip = z_;
    double *test_ip = z_test;

    float net_i[T_][H_];
    float net_f[T_][H_];
    float net_g[T_][H_];
    float net_o[T_][H_];

    float gate_i[T_][H_];
    float gate_f[T_][H_];
    float state_g[T_][H_];
    float gate_o[T_][H_];

    float h_prev[H_];
    float c_prev[H_];

    float c_state_diff[T_][H_];
    float g_gate_diff[T_][H_];
    float f_gate_diff[T_][H_];
    float i_gate_diff[T_][H_];
    float o_gate_diff[T_][H_];
    float h_gate_diff[T_][H_]; //cell output diff
    float h_i_gate_diff[T_][H_];
    float h_f_gate_diff[T_][H_];
    float h_g_gate_diff[T_][H_];
    float h_o_gate_diff[T_][H_];

    float didc[H_];
    float dfdc[H_];
    float dodc[H_];
    float didx[H_][I_];
    float dfdx[H_][I_];
    float dgdx[H_][I_];
    float dodx[H_][I_];

    float didh[H_][H_];
    float dfdh[H_][H_];
    float dgdh[H_][H_];
    float dodh[H_][H_];

    float didb[H_];
    float dfdb[H_];
    float dgdb[H_];
    float dodb[H_];

    float dzdh[K_][H_];
    float dzdb[K_];



    double error[T_][H_];
    double error_h[K_][H_];
    double gradient_O_h[T_][K_];

    double softmax_o[T_][K_];
    double softmax_loss[T_][K_];
    float loss;

    //double elapsed_time; //compute time of a iteration

    double sum = 0;

    // data

    int row = 19200;
    int col = 6;


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

        for (int j = 0; j < col; ++j)
        {
            input_train >> x[i][j];  //从输入流对象input读取字符到out
            input_test >> x_test[i][j];
            //cout << x_test[i][j] << " ";
            //x[i][j] = out[i][j];
            //count++;
        }
        //cout << " " << endl;
        //output << endl;
    }

    //cout << "count is 19200" << endl;
    //cout << count << endl;

    input_train.close();
    input_test.close();


    // init weights  weights initialize is ok
    for(int d = 0; d < H_; ++d)
    {

        //cell weights
        c_weight_i[d] = gaussian(0, 0.01);
        c_weight_f[d] = gaussian(0, 0.01);
        c_weight_o[d] = gaussian(0, 0.01);

        bias_i[d] = 0;
        bias_f[d] = 0;
        bias_g[d] = 0;
        bias_o[d] = 0;

        didc[d] = 0;
        dfdc[d] = 0;
        dodc[d] = 0;

        didb[d] = 0;
        dfdb[d] = 0;
        dgdb[d] = 0;
        dodb[d] = 0;

        for(int i = 0; i < I_; ++i)
        {
            weight_i[d][i] = gaussian(0, 0.01);
            weight_f[d][i] = gaussian(0, 0.01);
            weight_g[d][i] = gaussian(0, 0.01);
            weight_o[d][i] = gaussian(0, 0.01);

            didx[d][i] = 0;
            dfdx[d][i] = 0;
            dgdx[d][i] = 0;
            dodx[d][i] = 0;

        }

        for(int j = 0; j < H_; ++j)
        {
            h_weight_i[d][j] = gaussian(0, 0.01);
            h_weight_f[d][j] = gaussian(0, 0.01);
            h_weight_g[d][j] = gaussian(0, 0.01);
            h_weight_o[d][j] = gaussian(0, 0.01);

            didh[d][j] = 0;
            dfdh[d][j] = 0;
            dgdh[d][j] = 0;
            dodh[d][j] = 0;

        }
        //previous state
        for(int k = 0; k < K_; ++k)
        {
            h_z_weight[k][d] = gaussian(0, 0.01);
            dzdh[k][d] = 0;

        }
    }

// initialize h_prev[]

    for(int h = 0; h < H_; ++h)
    {
        h_prev[h] = 0;
        c_prev[h] = 0;

    }
// init bias_z
    for(int k =0; k < K_; ++k)
    {
	bias_z[k] = 0;
        dzdb[k] = 0;
    }
// init z_o[T_][K_] 初始化很重要

   for(int t = 0; t < T_; ++t)
   {
	for(int k = 0; k < K_; ++k)
	{
        z_o[t][k] = 0;
	z_o_test[t][k] = 0;
        gradient_O_h[t][k] = 0;


    }
	for(int d = 0; d < H_; ++d)
	{

		net_i[t][d] = 0;
		net_f[t][d] = 0;
		net_g[t][d] = 0;
		net_o[t][d] = 0;
        error[t][d] = 0;


        c_state_diff[t][d] = 0;
        g_gate_diff[t][d] = 0;
        f_gate_diff[t][d] = 0;
        i_gate_diff[t][d] = 0;
        o_gate_diff[t][d] = 0;

        h_gate_diff[t][d] = 0; //cell output diff
        h_i_gate_diff[t][d] = 0;
        h_f_gate_diff[t][d] = 0;
        h_g_gate_diff[t][d] = 0;
        h_o_gate_diff[t][d] = 0;

        //error[t][d] = 0;

	}

   }


    int max_epoch = 3000;

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
            batch_count ++;

            cout << n << " iteration" << endl;

            for(int t = 0; t < T_; ++t)  // T_ step a sample
            {
                //previous state

                for(int d = 0; d < H_; ++d)  // H_：hidden units
                {
                    // in, forget, g_sate, output
                    // compute like MLP
                    //cout << d << "hidden unit" << (n * T_ + t) << "x_input" << endl;
                    for(int i = 0; i < I_; ++i)
                    {

                        net_i[t][d] = weight_i[d][i] * x[id[n] * T_ + t][i] ;
                        net_f[t][d] = weight_f[d][i] * x[id[n] * T_ + t][i] ;
                        net_g[t][d] = weight_g[d][i] * x[id[n] * T_ + t][i] ;
                        net_o[t][d] = weight_o[d][i] * x[id[n] * T_ + t][i] ;

                        // test this block of code, x_input and bias are ok, weight_update is the problem
                    }

                    // !!! input is ok

                    // sum the h(t-1) state;  prev output的处理上应该存在问题

                    for(int j = 0; j < H_; ++j)
                    {

                        net_i[t][d] += h_weight_i[d][j] * h_prev[j];
                        net_f[t][d] += h_weight_f[d][j] * h_prev[j];
                        net_g[t][d] += h_weight_g[d][j] * h_prev[j];
                        net_o[t][d] += h_weight_o[d][j] * h_prev[j];
                        //std::cout << "h_prev" << h_prev[j] << "h_t" << h_t[t][j] << endl;
                    }
                    //sum c(t-1)
                    net_i[t][d] += c_weight_i[d] * c_prev[d];
                    net_f[t][d] += c_weight_f[d] * c_prev[d];

                    //compute input gate, forget gate, candiate cell state, output gate
                    gate_i[t][d] = sigmoid(net_i[t][d] + bias_i[d]);//input gate
                    gate_f[t][d] = sigmoid(net_f[t][d] + bias_f[d]);//forget gate
                    state_g[t][d] = tanh(net_g[t][d] + bias_g[d]);

                    //cell output
                    c_t[t][d] = gate_f[t][d] * c_prev[d] + gate_i[t][d] * state_g[t][d];

                    net_o[t][d] += c_weight_o[d] * c_t[t][d];

                    // output gate  多了一项c_t[t][d]
                    // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                    gate_o[t][d] = sigmoid(net_o[t][d] + bias_o[d]);//output gate

                    //cell net output one time step
                    h_t[t][d] = gate_o[t][d] * tanh(c_t[t][d]); //cell output
                    //cout << d << " h_t[t][d] " << h_t[t][d]<< endl;

                }

                //update prev h&c
                for(int d=0; d<H_; ++d)
                {
                    h_prev[d] = h_t[t][d];
                    c_prev[d] = c_t[t][d];
                }


            }
            // forward is ok  未出现 core dumped

            //output layer  计算完每一个输出神经元的softmax输出
            for(int t=0; t<T_; ++t)
            {
                //std::cout << t <<"step" << std::endl;

                for(int f = 0; f < K_; ++f)
                {
	            //if(f == 0) cout << "output to hidden: " << f << endl;
		    //if(f == 2) cout << "output to hidden: " << f << endl;
                    for(int h = 0; h < H_; ++h)
                    {
                        z_o[t][f] += h_z_weight[f][h] * h_t[t][h];

			//if(f == 0) cout << " h_z_weight: " << h_z_weight[f][h] << " ";
                        //if(f == 2) cout << " h_z_weight: " << h_z_weight[f][h] << " ";
                    }
                    z_[f] = z_o[t][f] + bias_z[f];// output value of every step
		    //cout << f << " value of ip before softmax: " << *ip << endl; //测试得softmax的输入存在问题，output unit 0的输入值很大
                    //cout << f << " bias_z: " << bias_z[f] << " z_o " << z_[f] << std::endl; //那么问题是 bias_z的值过大 导致输出神经元0的值过大
		    //ip++;                                                                //同时注意到，其他hidden to outputs的bias为0；
                    //at time step computesoftmax

                }
                softmax(z_);    //计算每一帧的softmax输出
                for(int f = 0; f < K_; ++f)  // output units 0 的概率为1 这是个问题,往上寻找问题的源头
                {

                    softmax_o[t][f] = *ip;
                    //sum += softmax_o[t][f];

                    //std::cout << "the probability of" << f << " output units:" << " "
                    //          << "lable: " << y[id[n]] << " "
                    //          << softmax_o[t][f] << std::endl;
                    ip++;
                    softmax_loss[t][f] = -log(softmax_o[t][f]);

                }
                //std::cout << "sum" << std::endl;
                //std::cout << sum << std::endl;
                //sum = 0;
                for(int k = 0; k < K_; ++k)
                {
                    loss += softmax_loss[t][k];

                }
                ip = z_;
                //num++;

            }
            //cout << "loss" << endl;
            //cout << loss << endl;

            //before backward is ok 程序可以执行到此处

            //Backward pass
            for( int t = T_-1; t >= 0; --t)
            {
                //compute error, T step , cell outputs
                for(int h = 0; h < H_; ++h)
                {
                    for(int k =0; k<K_; ++k)
                    {
                        if(y[id[n]] == k)
                        {
                            error[t][h] += h_z_weight[k][h] * (softmax_o[t][k] - 1.0);

                        }
                        else
                        {
                            error[t][h] += h_z_weight[k][h] * softmax_o[t][k];
                        }

                        //error_h[k][h] += error[t][h];

                   }

                    //cout << error[t][h] << " error[t][h] " << h << endl;
                }

                for(int k = 0; k<K_; ++k)
                {
                    if(y[id[n]] == k)
                    {
                        gradient_O_h[t][k] = softmax_o[t][k] - 1.0; //one problem should be 1 - softmax not y[k] - softmax;
                    }
                    else                                            //gradient 用来更新hidden to output的权重
                    {
                        gradient_O_h[t][k] = softmax_o[t][k];
                    }



                                                               // cout << k << " output units " << " gradient_O_h: " << gradient_O_h[t][k] << endl;
                }


                for(int d = 0; d < H_; ++d)
                {

                    //float error = h_t[t][d] - y[t][d];
                    //float dEdo = tanh(c_t[t][H_-d]);


                    //error += (z_o[t][k] - y[k]) * h_z_weight[K_][H_];

                    //float dEdc = gate_t[2*H_ + d] * (1 - pow(tanh(c_t[t][H_-d])),2);

                    //E[H_-d] = 0.5 * pow(error, 2);
                    //follow Graves



                    if( t==T_-1)
                    {
                        // cell state
                        // cell out put is error
                        //cell outputs
			 for(int hi = 0; hi < H_; ++hi)
                        {
                            h_i_gate_diff[t][d] += 0;
                            h_f_gate_diff[t][d] += 0;
                            h_g_gate_diff[t][d] += 0;
                            h_o_gate_diff[t][d] += 0;

                            didh[d][hi] += h_i_gate_diff[t][d];
                            dfdh[d][hi] += h_f_gate_diff[t][d];
                            dgdh[d][hi] += h_g_gate_diff[t][d];
                            dodh[d][hi] += h_g_gate_diff[t][d];
			}
                        h_gate_diff[t][d] = error[t][d] + h_i_gate_diff[t][d] + h_f_gate_diff[t][d] + h_g_gate_diff[t][d] + h_o_gate_diff[t][d];

                        //output gates
                        o_gate_diff[t][d] = dsigmoid(net_o[t][d]) * h_gate_diff[t][d] * tanh(c_t[t][d]);
                        //cell state
                        c_state_diff[t][d] = gate_o[t][d] * dtanh(c_t[t][d]) * h_gate_diff[t][d] +  c_weight_o[d]*o_gate_diff[t][d] ;
                        //candidate cell state
                        g_gate_diff[t][d] = gate_i[t][d] * dtanh(state_g[t][d]) * c_state_diff[t][d];
                        //forget gate
                        f_gate_diff[t][d] = dsigmoid(net_f[t][d]) * c_t[t-1][d] * c_state_diff[t][d];  //公式没有错误。。。
                        //input gate
                        i_gate_diff[t][d] = dsigmoid(net_i[t][d]) * c_state_diff[t][d] * state_g[t][d]; // input 有一个错误，dsigmoid应为net_i



                        // 改进该错误之后，c_weight 爆炸没有那么
                        // 快了 32sample 为nan

                    }
                    else
                    {
                        // T-1 then diff
                        //cell outputs
                        for(int hi = 0; hi < H_; ++hi)
                        {
                            h_i_gate_diff[t][d] += i_gate_diff[t+1][hi] * h_weight_i[d][hi];
                            h_f_gate_diff[t][d] += f_gate_diff[t+1][hi] * h_weight_f[d][hi];
                            h_g_gate_diff[t][d] += g_gate_diff[t+1][hi] * h_weight_g[d][hi];
                            h_o_gate_diff[t][d] += o_gate_diff[t+1][hi] * h_weight_o[d][hi];

                            didh[d][hi] += h_i_gate_diff[t][d];
                            dfdh[d][hi] += h_f_gate_diff[t][d];
                            dgdh[d][hi] += h_g_gate_diff[t][d];
                            dodh[d][hi] += h_g_gate_diff[t][d];

                        }
                        h_gate_diff[t][d] = error[t][d] + h_i_gate_diff[t][d] + h_f_gate_diff[t][d] + h_g_gate_diff[t][d] + h_o_gate_diff[t][d];
                        //output gate
                        o_gate_diff[t][d] = dsigmoid(net_o[t][d]) * h_gate_diff[t][d] * tanh(c_t[t][d]);
                        //cell state
                        c_state_diff[t][d] = gate_o[t][d] * dtanh(c_t[t][d]) * h_gate_diff[t][d] + gate_f[t+1][d] * c_state_diff[t+1][d] + c_weight_i[d] * i_gate_diff[t+1][d] + c_weight_f[d] * f_gate_diff[t+1][d] + c_weight_o[d] * o_gate_diff[t][d];
                        //candidate cell state
                        g_gate_diff[t][d] = gate_i[t][d] * dtanh(state_g[t][d]) * c_state_diff[t][d];
                        //forget gate
                        if(t == 0)
                       {
                           f_gate_diff[t][d] = 0;
                       }
                       else
                       {
                           f_gate_diff[t][d] = dsigmoid(net_f[t][d]) * c_t[t-1][d] * c_state_diff[t][d];
                       }
                        //input gate
                        i_gate_diff[t][d] = dsigmoid(net_i[t][d]) * c_state_diff[t][d] * state_g[t][d];


                    }

                    if(t==0)
                    {
                        didc[d] += 0;
                        dfdc[d] += 0;
                        dodc[d] += o_gate_diff[t][d] * c_t[t][d];
                    }
                    else
                    {
                        didc[d] += i_gate_diff[t][d] * c_t[t-1][d];
                        dfdc[d] += f_gate_diff[t][d] * c_t[t-1][d];
                        dodc[d] += o_gate_diff[t][d] * c_t[t][d];
                    }

                    for(int i = 0; i < I_; ++i)
                    {
                        didx[d][i] += i_gate_diff[t][d] * x[id[n] * T_ + t][i];
                        dfdx[d][i] += f_gate_diff[t][d] * x[id[n] * T_ + t][i];
                        dgdx[d][i] += g_gate_diff[t][d] * x[id[n] * T_ + t][i];
                        dodx[d][i] += o_gate_diff[t][d] * x[id[n] * T_ + t][i];
                    }

                    didb[d] += i_gate_diff[t][d];
                    dfdb[d] += f_gate_diff[t][d];
                    dgdb[d] += g_gate_diff[t][d];
                    dodb[d] += o_gate_diff[t][d];

                    for(int k = 0; k<K_; ++k)
                    {
                        dzdh[k][d] += gradient_O_h[t][k] * h_t[t][d];
                        dzdb[k] += gradient_O_h[t][k];

			 //cout << d << "hidden units: " << " dzdb[k] " << dzdb[k] <<endl;
                    }
		   

                }

            }

            for(int t_clear = 0; t_clear < T_; ++t_clear)
            {
           //cout << "clear"<< endl;

               for(int d = 0; d < H_; ++d)
               {
                   c_t[t_clear][d] = 0;
                   h_t[t_clear][d] = 0;
                   net_i[t_clear][d] = 0;
                   net_f[t_clear][d] = 0;
                   net_g[t_clear][d] = 0;
                   net_o[t_clear][d] = 0;
                   gate_i[t_clear][d] = 0;
                   gate_f[t_clear][d] = 0;
                   state_g[t_clear][d] = 0;
                   gate_o[t_clear][d] = 0;

                   // 梯度可以 在batch时累加
                   c_state_diff[t_clear][d] = 0;
                   g_gate_diff[t_clear][d] = 0;
                   f_gate_diff[t_clear][d] = 0;
                   i_gate_diff[t_clear][d] = 0;
                   o_gate_diff[t_clear][d] = 0;
                   h_gate_diff[t_clear][d] = 0; //cell output diff
                   h_i_gate_diff[t_clear][d] = 0;
                   h_f_gate_diff[t_clear][d] = 0;
                   h_g_gate_diff[t_clear][d] = 0;
                   h_o_gate_diff[t_clear][d] = 0;
                   error[t_clear][d] = 0;

                   h_prev[d] = 0;
                   c_prev[d] = 0;
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


            //batch_count++;
            // before update is ok （core dumped problem）

            if(batch_count == batch_size) // version of batch = 1
            {

                batch_count = 0;

                // cout << n << "sample:" << " " <<endl ;
                for(int d = 0; d < H_; ++d)  //problem is here(core dumped)
                {
                    // cout << t << "steps:" << " " <<endl ;


                        c_weight_i[d] = c_weight_i[d] - lr * (1/batch_size) * didc[d];
		        c_weight_f[d] = c_weight_f[d] - lr * (1/batch_size) * dfdc[d];
		        c_weight_o[d] = c_weight_o[d] - lr * (1/batch_size) * dodc[d];

                        //if(n == 62) cout << "n  in update "  << endl;
                        // if(n == 63) cout << "n  in update  the problem"  << endl;

                        for(int i = 0; i < I_; ++i)
                        {
                              weight_i[d][i] = weight_i[d][i] - lr * (1/batch_size) * didx[d][i];
                              weight_f[d][i] = weight_f[d][i] - lr * (1/batch_size) * dfdx[d][i];
                              weight_g[d][i] = weight_g[d][i] - lr * (1/batch_size) * dgdx[d][i];
                              weight_o[d][i] = weight_o[d][i] - lr * (1/batch_size) * dodx[d][i];

                        }
                        //update weights of hidden to hidden

                        //if(n == 63) cout << "n  in update  the problem"  << endl;

                        //if(t < T_-1)  // one problem for update weights
                        //{

                            for(int h = 0; h<H_; ++h)
                            {
                                h_weight_i[d][h] = h_weight_i[d][h] - lr * (1/batch_size) * didh[d][h];
                                h_weight_f[d][h] = h_weight_f[d][h] - lr * (1/batch_size) * dfdh[d][h];
                                h_weight_g[d][h] = h_weight_g[d][h] - lr * (1/batch_size) * dgdh[d][h];
                                h_weight_o[d][h] = h_weight_o[d][h] - lr * (1/batch_size) * dodh[d][h];

                                //cout << "h_weight_i:" << " " << h_weight_i[d][h] << " "
                                //<< "h_weight_f:" << " " << h_weight_f[d][h] << " "
                                //<< "h_weight_g:" << " " << h_weight_g[d][h] << " "
                                //<< "h_weight_o:" << " " << h_weight_o[d][h] << endl;

                            }
                        //}

                        //hidden to output weights update error
                        for(int k = 0; k<K_; ++k)
                        {
                            h_z_weight[k][d] = h_z_weight[k][d] - lr * (1/batch_size) * dzdh[k][d];   // update equation is one of the error
                            //cout << "h_z_weight:" << " " << h_z_weight[k][d] << endl;
                        }


                       // bias of hidden to output
                        bias_i[d] = bias_i[d] - lr * (1/batch_size) * didb[d];
                        bias_f[d] = bias_f[d] - lr * (1/batch_size) * dfdb[d];
                        bias_g[d] = bias_g[d] - lr * (1/batch_size) * dgdb[d];
                        bias_o[d] = bias_o[d] - lr * (1/batch_size) * dodb[d];

                       // test predict accuary every batch

                  }

		  for(int k =0; k < K_; ++k)  bias_z[k] = bias_z[k] - lr * (1/batch_size) * dzdb[k]; //update bias_z




                for(int test_iter = 0 ; test_iter < 1; ++test_iter)
                {
                    cout << "test iter" << endl;
                    double test_accuary = 0.0;

                    for(int sample = 0; sample < N_; ++sample)
                    {

                        for(int t = 0; t < T_; ++t)  // T_ step
                        {
                            //previous state

                            for(int d = 0; d < H_; ++d)  // H_：hidden units
                            {
                                // in, forget, g_sate, output
                                // compute like MLP
                                for(int i = 0; i < I_; ++i)
                                {
                                    net_i[t][d] = weight_i[d][i] * x_test[sample * T_ + t][i];
                                    net_f[t][d] = weight_f[d][i] * x_test[sample * T_ + t][i];
                                    net_g[t][d] = weight_g[d][i] * x_test[sample * T_ + t][i];
                                    net_o[t][d] = weight_o[d][i] * x_test[sample * T_ + t][i];
                                // test this block of code, x_input and bias are ok, weight_update is the problem
                                }
                                //sum the h(t-1) state;
                                for(int j = 0; j < H_; ++j)
                                {
                                    net_i[t][d] += h_weight_i[d][j] * h_prev[j];
                                    net_f[t][d] += h_weight_f[d][j] * h_prev[j];
                                    net_g[t][d] += h_weight_g[d][j] * h_prev[j];
                                    net_o[t][d] += h_weight_o[d][j] * h_prev[j];
                                }
                                //sum c(t-1)
                                net_i[t][d] += c_weight_i[d] * c_prev[d];
                                net_f[t][d] += c_weight_f[d] * c_prev[d];

                                //compute input gate, forget gate, candiate cell state, output gate
                                gate_i[t][d] = sigmoid(net_i[t][d] + bias_i[d]);
                                gate_f[t][d] = sigmoid(net_f[t][d] + bias_f[d]);
                                state_g[t][d] = tanh(net_g[t][d] + bias_g[d]);
                                //cell output

                                c_t[t][d] = gate_f[t][d] * c_prev[d] + gate_i[t][d] * state_g[t][d];
                                net_o[t][d] += c_weight_o[d] * c_t[t][d];

                                // output gate  多了一项c_t[t][d]
                                // gate_o[t][d] = sigmoid(net_o[t][d] + c_t[t][d]);
                                gate_o[t][d] = sigmoid(net_o[t][d] + bias_o[d]);
                                //cell netoutput one time step
                                h_t[t][d] = gate_o[t][d] * tanh(c_t[t][d]);
				//std::cout << "h_t[t][d] " << h_t[t][d] << endl;

                            }
                            //update prev h&c
                          for(int d=0; d<H_; ++d)
                            {
                                h_prev[d] = h_t[t][d];
                                c_prev[d] = c_t[t][d];

                            }



                        }

                        //compute accuary
                        //after a sample clear h_prev c_prev

                        for(int h = 0; h < H_; ++h)
                        {
                            h_prev[h] = 0;
                            c_prev[h] = 0;
                        }

                        for(int t=0; t<T_; ++t)
                        {
                            //std::cout << t <<"step" << std::endl;
                            for(int f = 0; f < K_; ++f)

                            {
                                for(int h = 0; h < H_; ++h)
                                {
                                    z_o_test[t][f] += h_z_weight[f][h] * h_t[t][h];

                                }
                                //z_o_test[t][f] = z_o_test[t][f] + bias_z[f];
                                z_test[f] = z_o_test[t][f] + bias_z[f];;
				//cout << "z_test[f] " << z_test[f] << endl;
                                //at time step computesoftmax
                            }
                            softmax(z_test);
                            for(int f = 0; f < K_; ++f)
                            {

                                softmax_o[t][f] = *test_ip;
                                sum += softmax_o[t][f];
                                //std::cout << "the probability of test " << f << " output units "
				//	  << "label: " << y_test[sample] << " "
                                //          << softmax_o[t][f] << std::endl;
                                test_ip++;
                                //std::cout << ip << std::endl;
                                softmax_loss[t][f] = -log(softmax_o[t][f]);

                            }
                            //std::cout << "sum" << std::endl;
                            //std::cout << sum << std::endl;
                            sum = 0;

                            for(int k = 0; k < K_; ++k)
                            {
                                loss += softmax_loss[t][k];

                            }
                            test_ip = z_test;


                        }

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

                                }


                            } //compute the
                            //cout << "y_predict " << y_predict << "y_test[sample] " << y_test[sample] <<endl;
                            if(y_predict == y_test[sample])
                                {
                                    test_accuary += 1 / double(300 * N_ );
                                    //cout << "accuary change ：" << test_accuary << endl;
                                }

                            //cout << "sample :" << sample << " "
                            //    << "y_predict :" << y_predict << " "
                            //    << "y_test[sample] :" << y_test[sample] <<endl;

                        }
                         for(int t_clear = 0; t_clear < T_; ++t_clear)
                {

                                            for(int d = 0; d < H_; ++d)
                    {
                        c_t[t_clear][d] = 0;
                        h_t[t_clear][d] = 0;
                        net_i[t_clear][d] = 0;
                        net_f[t_clear][d] = 0;
                        net_g[t_clear][d] = 0;
                        net_o[t_clear][d] = 0;
                        gate_i[t_clear][d] = 0;
                        gate_f[t_clear][d] = 0;
                        state_g[t_clear][d] = 0;
                        gate_o[t_clear][d] = 0;

                        // 梯度可以 在batch时累加


                    }

                    for(int k = 0; k < K_; ++k)
                    {
                        z_o[t_clear][k] = 0 ;
                        z_o_test[t_clear][k] = 0 ;
                        gradient_O_h[t_clear][k] = 0;
                        softmax_o[t_clear][k] = 0;
                        softmax_loss[t_clear][k] = 0;
                        z_[k] = 0;


                    }
                }

                 }

                    cout << "test_accuary :" << test_accuary << endl;

                }


                 for(int t_clear = 0; t_clear < T_; ++t_clear)
                {
                //cout << "clear"<< endl;

                    for(int d = 0; d < H_; ++d)
                    {
                        c_t[t_clear][d] = 0;
                        h_t[t_clear][d] = 0;
                        net_i[t_clear][d] = 0;
                        net_f[t_clear][d] = 0;
                        net_g[t_clear][d] = 0;
                        net_o[t_clear][d] = 0;
                        gate_i[t_clear][d] = 0;
                        gate_f[t_clear][d] = 0;
                        state_g[t_clear][d] = 0;
                        gate_o[t_clear][d] = 0;

                        // 梯度可以 在batch时累加

                        error[t_clear][d] = 0;


                    }

                    for(int k = 0; k < K_; ++k)
                    {
                        z_o[t_clear][k] = 0 ;
                        gradient_O_h[t_clear][k] = 0;
                        softmax_o[t_clear][k] = 0;
                        softmax_loss[t_clear][k] = 0;
                        z_[k] = 0;


                    }

                 }

                 for(int d = 0; d < H_; ++d)
                 {

                     didc[d] = 0;
                     dfdc[d] = 0;
                     dodc[d] = 0;

                     didb[d] = 0;
                     dfdb[d] = 0;
                     dgdb[d] = 0;
                     dodb[d] = 0;


                     h_prev[d] = 0;
                     c_prev[d] = 0;
                     for(int i =0; i < I_; ++i)
                     {
                         didx[d][i] = 0;
                         dfdx[d][i] = 0;
                         dgdx[d][i] = 0;
                         dodx[d][i] = 0;
                     }
                     for(int j = 0; j < H_; ++j)
                     {
                         didh[d][j] = 0;
                         dfdh[d][j] = 0;
                         dgdh[d][j] = 0;
                         dodh[d][j] = 0;
                     }
                     for(int k = 0; k<K_; ++k)
                     {
                         dzdh[k][d] = 0;
                         dzdb[k] = 0;//
                     }

                 }


             }
                //clear end

            }





    }


    return 0;
}







