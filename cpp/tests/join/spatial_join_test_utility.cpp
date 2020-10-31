#include <time.h>
#include <sys/time.h>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "spatial_join_test_utility.hpp"

//placeholder for structus and functions needed to run NYC taxi experiments
//they will be incorporated into the new io module of cuspatial in a later release

//=================================================================================================
struct rec_cnyc
{
    int xf,yf,xt,yt;
};

float calc_time(const char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}


void gen_rand_idx(std::vector<uint32_t>& indices,uint32_t num_counts, uint32_t num_samples)
{
    if(num_samples<num_counts)
    {
        std::seed_seq seed{time(0)};
        std::mt19937 g(seed);
        std::uniform_int_distribution<> dist_rand (0,num_counts-1);
        indices.resize(num_samples);
        std::generate(indices.begin(), indices.end(), [&] () mutable { return dist_rand(g); });
    }
    else if(num_samples==num_counts)
    {
        indices.resize(num_counts);
        std::generate(indices.begin(), indices.end(), [n = 0] () mutable { return n++; });
    }
    else
             std::cout<<"num_samples="<<num_samples<<" must be less or equal to num_counts="<<num_counts;
   assert(indices!=nullptr);
}

size_t read_point_binary(const char *fn,double*& h_pnt_x,double*& h_pnt_y)
{
    FILE *fp=nullptr;
    if((fp=fopen(fn,"rb"))==nullptr)
    {
        std::cout<<"Can not open file "<<fn<<" for reading"<<std::endl;
        exit(-1);
    }
    fseek (fp , 0 , SEEK_END);
    size_t sz=ftell (fp);
    assert(sz%sizeof(rec_cnyc)==0);
    size_t num_rec = sz/sizeof(rec_cnyc);
    std::cout<<"num_rec="<<num_rec<<std::endl;
    fseek (fp , 0 , SEEK_SET);

    h_pnt_x=new double[num_rec];
    h_pnt_y=new double[num_rec];
    assert(h_pnt_x!=nullptr && h_pnt_y!=nullptr);
    struct rec_cnyc *temp=new rec_cnyc[num_rec];

    size_t t=fread(temp,sizeof(rec_cnyc),num_rec,fp);
    if(t!=num_rec)
    {
        std::cout<<"cny coord read error: expected="<<num_rec<<", actual="<<t<<std::endl;
        exit(-1);
    }
    for(uint32_t i=0;i<num_rec;i++)
    {
        h_pnt_x[i]=temp[i].xf;
        h_pnt_y[i]=temp[i].yf;
    }
    fclose(fp);
    delete[] temp;
    return num_rec;
    //timeval t0,t1;
    //gettimeofday(&t0, nullptr);
}
