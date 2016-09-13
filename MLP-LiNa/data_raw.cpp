#include<iostream>
#include<stdio.h>
using namespace std;
int main(){
    FILE* fp;
int max_line_num=569;
int max_row_num=30;
    fp=fopen("/home/lina/programme/uci.txt","r");
    long int a;
    char b;
    char label;
    float d[max_line_num][30];
    int c[569];
    for(int i=0;i<569;i++)
    {
        fscanf(fp,"%ld",&a);
        fscanf(fp,"%c",&b);
        fscanf(fp,"%c",&label);
        if(label=='M')
            c[i]=1;
        else
            c[i]=0;
        //fscanf(fp,"%c",&b);
        for(int j=0;j<30;j++)
        {
            fscanf(fp,"%c%f",,&b,&d[i][j]);
            //fscanf(fp,"%c",&b);
        }
        //fscanf(fp,"%f",&d[i][29]);
    }
      fclose(fp);


      FILE* ff;
      ff=fopen("/home/lina/programme/data1","w");
      for(int i=0;i<569;i++){
          for(int j=0;j<30;j++){
              fprintf(ff,"%lf\t",d[i][j]);
          }
          fprintf(ff,"\n");
      }
      fclose(ff);
     FILE *f1;
      f1=fopen("/home/lina/programme/data1","r");
      float data[569][30];
      for (int i=0;i<569;i++){
          for(int j=0;j<30;j++){
              fscanf(f1,"%f",&data[i][j]);
          }
      }
      fclose(f1);


      float max[30];
      float min[30];
      for(int i=0;i<30;i++){
          max[i]=-9999;
          min[i]=9999;
      }
      float data2[30][569];
      for(int i=0;i<30;i++){
          for(int j=0;j<569;j++){
              data2[i][j]=data[j][i];
        }
      }
      for(int i=0;i<30;i++){
          for(int j=0;j<569;j++){
              if(max[i]<data2[i][j])
                 max[i]=data2[i][j];
              if(min[i]>data2[i][j])
                 min[i]=data2[i][j];
          }
      }
      for(int i=0;i<30;i++){
          for(int j=0;j<569;j++)
              data2[i][j]=(data2[i][j]-min[i])/(max[i]-min[i]);
      }

       FILE* nf;
       nf=fopen("/home/lina/programme/new","w");
       for(int i=0;i<569;i++){
           for(int j=0;j<30;j++){
               data[i][j]=data2[j][i];
               fprintf(nf,"%f\t",data[i][j]);
           }
           fprintf(nf,"\n");
       }
      fclose(nf);
       return 0;
      }
