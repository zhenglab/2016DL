//思路：剔除后重新排序

#include<iostream>
using namespace std;
int main()
{
    int n,m,i,j,a;
    cout<<"please input the number of digit: ";
    cin>>a;
    cin>>m;
    n=a;
    //n=8;
    //m=3;
    int table[n]={0};
    int daiti[n]={0};
    //int *table;
    //table=new int[n];
    for(i=0;i<n;i++)
    {
        table[i]=i+1;
        //cout<<table[i];
    }

    for(n=a;n>m;n--)
    {
        for(i=n-m,j=0;i<(n-1);i++,j++)
        {
            daiti[n-m+j]=table[i-n+m];
        }

        for(i=0;i<(n-m);i++)
           {
                table[i]=table[i+m];
           }
        table[n-m]=daiti[n-m];

    }
    cout<<table[0];
   // delete table;
    return 0;
}

