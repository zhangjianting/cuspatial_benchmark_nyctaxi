#include<sys/time.h>
#include<time.h>

#include <ogrsf_frmts.h>
#include <geos_c.h>

//internal helper function defintions, documentation TBD

int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
        std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v,
        uint8_t type, std::vector<OGRGeometry *>& polygon_vec, std::vector<uint32_t>& idx_vec);

void write_shapefile(const char * file_name,uint32_t num_poly,
    const double *x1,const double * y1,const double *x2,const double * y2);

void polyvec_to_bbox(const std::vector<OGRGeometry *>& h_ogr_polygon_vec,const char * file_name,
    double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2);

void rand_points_ogr_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const std::vector<OGRGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices);

void matched_pairs_ogr_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const uint32_t *h_pq_quad_idx,  const uint32_t *h_pq_poly_idx,
    const uint32_t *h_qt_length,  const uint32_t * h_qt_fpos,
    const std::vector<OGRGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices);

void rand_points_geos_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const std::vector<GEOSGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices);

void matched_pairs_geos_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const uint32_t *h_pq_quad_idx,  const uint32_t *h_pq_poly_idx,
    const uint32_t *h_qt_length,  const uint32_t * h_qt_fpos,
    const std::vector<GEOSGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices);

