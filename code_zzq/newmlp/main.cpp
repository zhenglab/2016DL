#include <QCoreApplication>
#include<vector>
#include<time.h>
#include<fstream>
#include<string>
using namespace std;
double getrand(){
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if (phase == 0) {
        do {
            double U1 = (double) rand() / RAND_MAX;
            double U2 = (double) rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    double x=X*0.1;
    return x;
}
//sigmoid函数
inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
//sigmoid的导函数
inline double fsigmoid(double x){
    return x*(1-x);
}

class layer{
public:
    int ns;
    int num_node;
    int samth;
    double **forward_weight;
    double **backward_weight;
    double **dfw;
    double **dbw;
    double **bsdfw;
    double **bsdbw;
    int set_nn(int m){
        num_node=m;
        return 0;
    }
    int set_samth(int m){
        samth=m;
        return 0;
    }
    int get_samth(){
        return samth;
    }
};

class Ilayer:public layer{
public:
    Ilayer(int ns,int nn){
        this->ns=ns;
        this->num_node=nn;
        ifstream ifile;
        ifile.open("data");
        double **p;
        p=new double*[ns];
        for(int i=0;i<ns;i++){
            p[i]=new double[nn];
            for(int j=0;j<nn;j++){
                ifile>>p[i][j];
            }
        }
    }

};
class Hlayer:public layer{
public:
    double *ym;
    double *yn;
    double *bias;
    double *dbias;
    Hlayer(int nn){
        this->num_node=nn;
        ym=new double[nn];
        yn=new double[nn];
        bias=new double[nn];
        dbias=new double[nn];
        for(int i=0;i<nn;++i){
            ym[i]=0;
            yn[i]=0;
            bias[i]=getrand();
            dbias[i]=0;
        }
    }
};
class Olayer:public layer{
public:
    double *yflag;
    double *yout;
    int label;
    int *all_label;
    Olayer(int ns,int nn){
        this->ns=ns;
        this->num_node=nn;
        yflag=new double[nn];
        yout=new double[nn];
        all_label=new int[ns];
        ifstream ifile;
        ifile.open("label");
        for(int i=0;i<ns;++i){
            ifile>>all_label[i];
        }
        ifile.close();
        for(int i=0;i<nn;++i){
            yflag[i]=0;
            yout[i]=0;
        }
        label=all_label[samth];
        yflag[label]=1;
    }
};

void connection(layer *l1,layer *l2){
    int m=l1->num_node;
    int n=l2->num_node;
    l1->forward_weight=new double*[n];
    for(int i=0;i<n;++i){
        l1->forward_weight[i]=new double[m];
        for(int j=0;j<m;++j){
            l1->forward_weight[i][j]=getrand();
        }
    }
    l2->backward_weight=new double*[m];
    for(int i=0;i<m;++i){
        l2->backward_weight[i]=new double[n];
        for(int j=0;j<n;j++){
            l2->backward_weight[i][j]=getrand();
        }
    }
}
void layer_forward(Ilayer *input,Hlayer *hidden,Olayer *output){
    input->
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    return a.exec();
}
