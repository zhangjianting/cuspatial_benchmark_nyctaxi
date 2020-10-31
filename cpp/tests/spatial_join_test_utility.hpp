#pragma once

#include<sys/time.h>
#include<time.h>
#include<vector>


float calc_time(const char *msg,timeval t0, timeval t1);

void gen_rand_idx(std::vector<uint32_t>& indices,uint32_t num_counts, uint32_t num_samples);

size_t read_point_binary(const char *fn,double*& h_pnt_x,double*& h_pnt_y);

bool compute_mismatch(uint32_t num_pp_pairs,const std::vector<uint32_t>&  h_org_poly_idx_vec,
    const uint32_t *h_pnt_search_idx, const std::vector<uint32_t>& h_pnt_len_vec,const uint32_t * h_poly_search_idx,
    const uint32_t * h_pp_pnt_idx,const uint32_t *h_pp_poly_idx,
    const double *h_pnt_x, const double * h_pnt_y);

