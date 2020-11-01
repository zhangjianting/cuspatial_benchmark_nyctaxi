#include <stdint.h>
#include <iostream>
#include "spatial_join_test_utility.hpp"
#include "spatial_join_geo_utility.hpp"

struct SpatialJoinNYCTaxiVerify
{
    uint32_t num_pnts=0;

    uint32_t num_quadrants=0;

    uint32_t num_pq_pairs=0;

    uint32_t num_pp_pairs=0;

     //point x/y on host
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;
    uint32_t *h_point_indices=nullptr;

    uint32_t num_poly=0,num_ring=0,num_vertex=0;

    //polygon vertices x/y
    double *h_poly_x=nullptr,*h_poly_y=nullptr;

    //quadtree length/fpos
    uint32_t *h_qt_length=nullptr,*h_qt_fpos=nullptr;

    //quadrant/polygon pairs
    uint32_t *h_pq_quad_idx=nullptr,*h_pq_poly_idx=nullptr;

    uint32_t *h_pp_pnt_idx=nullptr,*h_pp_poly_idx=nullptr;

    //poygons using GDAL/OGR OGRGeometry structure
    std::vector<OGRGeometry *> h_ogr_polygon_vec;
    std::vector<GEOSGeometry *> h_geos_polygon_vec;

    //sequential idx 0..num_poly-1 to index h_ogr_polygon_vec
    //needed when actual polygons in spatial join are only a subset, e.g., multi-polygons only
    std::vector<uint32_t> h_org_poly_idx_vec;

    //point idx that intersect with at least one polygon based on GDAL/OGR OGRGeometry.Contains
    std::vector<uint32_t> h_pnt_idx_vec;

    //# of poylgons that are contain points indexed by h_pnt_idx_vec at the same index
    std::vector<uint32_t> h_pnt_len_vec;

    //#polygon indices for those contain points in h_pnt_idx_vec; sequentially concatenated
    std::vector<uint32_t> h_poly_idx_vec;

    void setup_polygons(const char *file_name)
    {
        SpatialJoinNYCTaxiVerify test;

        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();
        GDALDatasetH hDS = GDALOpenEx(file_name, GDAL_OF_VECTOR, nullptr, nullptr, nullptr );
        if(hDS==nullptr)
        {
            std::cout<<"Failed to open ESRI Shapefile dataset "<< file_name<<std::endl;
            exit(-1);
        }
        //a shapefile abstracted as a GDALDatasetGetLayer typically has only one layer
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );

        test.h_ogr_polygon_vec.clear();
        test.h_geos_polygon_vec.clear();
        test.h_org_poly_idx_vec.clear();

        //type: 0 for all, 1 for simple polygons and 2 for multi-polygons
        uint8_t type=0;
        uint32_t num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v,type,h_ogr_polygon_vec,h_org_poly_idx_vec);
        assert(num_f>0);

        h_geos_polygon_vec.clear();
        GEOSContextHandle_t hGEOSCtxt = OGRGeometry::createGEOSContext();
        for(uint32_t i=0;i<num_f;i++)
        {
            OGRGeometry *poOGRPoly=h_ogr_polygon_vec[i];
            GEOSGeometry *poGEOSPoly = poOGRPoly->exportToGEOS(hGEOSCtxt);
            h_geos_polygon_vec.push_back(poGEOSPoly);
        }

        //num_group=g_len_v.size();
        test.num_poly=f_len_v.size();
        test.num_ring=r_len_v.size();
        test.num_vertex=x_v.size();

        uint32_t *h_poly_flen=new uint32_t[num_poly];
        uint32_t *h_poly_rlen=new uint32_t[num_ring];
        assert(h_poly_flen!=nullptr && h_poly_rlen!=nullptr);

        test.h_poly_x=new double [num_vertex];
        test.h_poly_y=new double [num_vertex];
        assert(h_poly_x!=nullptr && h_poly_y!=nullptr);

        std::copy_n(f_len_v.begin(),num_poly,h_poly_flen);
        std::copy_n(r_len_v.begin(),num_ring,h_poly_rlen);
        std::copy_n(x_v.begin(),num_vertex,h_poly_x);
        std::copy_n(y_v.begin(),num_vertex,h_poly_y);
        std::cout<<"setup_polygons: num_poly="<<num_poly<<" num_ring="<<num_ring<<" num_vertex="<<num_vertex<<std::endl;

        //note that the bbox of all polygons will used as the Area of Intersects (AOI) to join points with polygons
        double x1=*(std::min_element(x_v.begin(),x_v.end()));
        double x2=*(std::max_element(x_v.begin(),x_v.end()));
        double y1=*(std::min_element(y_v.begin(),y_v.end()));
        double y2=*(std::max_element(y_v.begin(),y_v.end()));
        std::cout<<"read_polygon_bbox: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;
    }

    void compare_random_points(uint32_t num_samples,uint32_t num_print_interval,bool using_geos)
    {
        std::cout<<"compare_random_points: num_quadrants="<<test.num_quadrants
            <<" num_pp_pair="<<test.num_pp_pairs<<" num_samples="<<num_samples<<std::endl;

        std::vector<uint32_t> rand_indices;
        //gen_rand_idx(rand_indices,test.num_pnts,num_samples);
        for(int i=0;i<test.num_pnts;i++)
            rand_indices.push_back(i);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);

        //h_pnt_idx_vec, h_pnt_len_vec and h_poly_idx_vec will be cleared first

        if(using_geos)
        {
            rand_points_geos_pip_test(num_print_interval,rand_indices, test.h_geos_polygon_vec,test.h_pnt_idx_vec,
                test.h_pnt_len_vec,test.h_poly_idx_vec,test.h_pnt_x,test.h_pnt_y,test.h_point_indices);
        }
        else
        {
            rand_points_ogr_pip_test(num_print_interval,rand_indices, test.h_ogr_polygon_vec,test.h_pnt_idx_vec,
                test.h_pnt_len_vec,test.h_poly_idx_vec,test.h_pnt_x,test.h_pnt_y,test.h_point_indices);
         }
        gettimeofday(&t1, nullptr);
        float cpu_time=calc_time("cpu random sampling computing time = ",t0,t1);
    }

    void compare_matched_pairs(uint32_t num_samples,uint32_t num_print_interval,bool using_geos)
    {
        std::cout<<"compare_matched_pairs: num_quadrants="<<test.num_quadrants<<" num_pq_pairs"<<test.num_pq_pairs
            <<" num_pp_pair="<<test.num_pp_pairs<<" num_samples="<<num_samples<<std::endl;

        std::vector<uint32_t> rand_indices;
        gen_rand_idx(rand_indices,test.num_pq_pairs,num_samples);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);

        if(using_geos)
        {
            matched_pairs_geos_pip_test(num_print_interval,rand_indices,
                test.h_pq_quad_idx,test.h_pq_poly_idx,test.h_qt_length,test.h_qt_fpos,
                test.h_geos_polygon_vec,test.h_pnt_idx_vec,test.h_pnt_len_vec,test.h_poly_idx_vec,
                test.h_pnt_x,test.h_pnt_y,test.h_point_indices);
        }
        else
        {
            matched_pairs_ogr_pip_test(num_print_interval,rand_indices,
                test.h_pq_quad_idx,test.h_pq_poly_idx,test.h_qt_length,test.h_qt_fpos,
                test.h_ogr_polygon_vec,test.h_pnt_idx_vec,test.h_pnt_len_vec,test.h_poly_idx_vec,
                test.h_pnt_x,test.h_pnt_y,test.h_point_indices);

        }
        gettimeofday(&t1, nullptr);
        float cpu_time=calc_time("cpu matched-pair computing time",t0,t1);
    }

    void read_points_bin(const char *file_name)
    {
        CUDF_EXPECTS(file_name!=NULL,"file_name can not be NULL");
        FILE *fp=fopen(file_name,"rb");
        CUDF_EXPECTS(fp!=NULL, "can not open file for input");
        CUDF_EXPECTS(fread(&(test.num_pnts),sizeof(uint32_t),1,fp)==1,"reading num_pnt failed");
        CUDF_EXPECTS(fread(&(test.num_quadrants),sizeof(uint32_t),1,fp)==1,"reading num_quadrants failed");
        CUDF_EXPECTS(fread(&(test.num_pq_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pq_pairs failed");
        CUDF_EXPECTS(fread(&(test.num_pp_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pp_pairs failed");
        std::cout<<"num_pnts="<<test.num_pnts<<" num_quadrants="<<test.num_quadrants<<" num_pq_pairs="<<test.num_pq_pairs<<" num_pp_pairs="<<test.num_pp_pairs<<std::endl;

        std::cout<<"reading points..."<<std::endl;
        test.h_pnt_x=new double[test.num_pnts];
        test.h_pnt_y=new double[test.num_pnts];
        test.h_point_indices= new uint32_t[test.num_pnts];
        CUDF_EXPECTS( test.h_pnt_x!=NULL && test.h_pnt_y!=NULL && test.h_point_indices!=NULL	 ,"allocating memory for points on host failed");

        CUDF_EXPECTS(fread(test.h_pnt_x,sizeof(double),test.num_pnts,fp)==test.num_pnts,"reading h_pnt_x failed");
        CUDF_EXPECTS(fread(test.h_pnt_y,sizeof(double),test.num_pnts,fp)==test.num_pnts,"reading h_pnt_y failed");
        CUDF_EXPECTS(fread(test.h_point_indices,sizeof(uint32_t),test.num_pnts,fp)==test.num_pnts,"reading h_point_indices failed");

        thrust::copy(test.h_pnt_x,test.h_pnt_x+test.num_pnts,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;
        thrust::copy(test.h_pnt_y,test.h_pnt_y+test.num_pnts,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;
        thrust::copy(test.h_point_indices,test.h_point_indices+test.num_pnts,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

        std::cout<<"reading quadrants..."<<std::endl;
        test.h_qt_length=new uint32_t[test.num_quadrants];
        test.h_qt_fpos=new uint32_t[test.num_quadrants];
        CUDF_EXPECTS( test.h_qt_length!=NULL && test.h_qt_fpos!=NULL,"allocating memory for quadrants on host failed");

        CUDF_EXPECTS(fread(test.h_qt_length,sizeof(uint32_t),test.num_quadrants,fp)==test.num_quadrants,"reading h_qt_length failed");
        CUDF_EXPECTS(fread(test.h_qt_fpos,sizeof(uint32_t),test.num_quadrants,fp)==test.num_quadrants,"reading h_qt_fpos failed");

        std::cout<<"reading quadrant/polygon pairs..."<<std::endl;
        test.h_pq_quad_idx=new uint32_t[test.num_pq_pairs];
        test.h_pq_poly_idx=new uint32_t[test.num_pq_pairs];
        CUDF_EXPECTS( test.h_pq_poly_idx!=NULL && test.h_pq_quad_idx!=NULL,"allocating memory for quadrant-polygon pairs on host failed");

        CUDF_EXPECTS(fread(test.h_pq_quad_idx,sizeof(uint32_t),test.num_pq_pairs,fp)==test.num_pq_pairs,"reading h_pq_quad_idx failed");
        CUDF_EXPECTS(fread(test.h_pq_poly_idx,sizeof(uint32_t),test.num_pq_pairs,fp)==test.num_pq_pairs,"reading h_pq_poly_idx failed");

        std::cout<<"reading point/polygon pairs..."<<std::endl;
        test.h_pp_poly_idx=new uint32_t[test.num_pp_pairs];
        test.h_pp_pnt_idx=new uint32_t[test.num_pp_pairs];
        CUDF_EXPECTS(test.h_pp_poly_idx!=NULL && test.h_pp_pnt_idx!=NULL,"allocating memory for point-polygon pairs on host failed");

        CUDF_EXPECTS(fread(test.h_pp_poly_idx,sizeof(uint32_t),test.num_pp_pairs,fp)==test.num_pp_pairs,"reading h_pp_poly_idx failed");
        CUDF_EXPECTS(fread(test.h_pp_pnt_idx,sizeof(uint32_t),test.num_pp_pairs,fp)==test.num_pp_pairs,"reading h_pp_pnt_idx failed");

        for(int i=0;i<test.num_pp_pairs;i++)
        {
           uint32_t pid=test.h_point_indices[test.h_pp_pnt_idx[i]];
           printf("%d,%10.5f, %10.5f, %d\n",i,test.h_pnt_x[pid],test.h_pnt_y[pid],test.h_pp_poly_idx[i]);
	   }

    }

    void tear_down()
    {
        delete[] test.h_poly_x; test.h_poly_x=nullptr;
        delete[] test.h_poly_y; test.h_poly_y=nullptr;

        delete[] test.h_pnt_x; test.h_pnt_x=nullptr;
        delete[] h_pnt_y; h_pnt_y=nullptr;

        delete[] test.h_pq_quad_idx; test.h_pq_quad_idx=nullptr;
        delete[] h_pq_poly_idx; h_pq_poly_idx=nullptr;

        delete[] test.h_qt_length; test.h_qt_length=nullptr;
        delete[] test.h_qt_fpos; test.h_qt_fpos=nullptr;
    }

};

/*
 * There could be multple configureations (minior ones are inside parentheses):
 * pick one of three polygon datasets
 * choose from compare_random_points and compare_matched_pairs
*/

int main(){

    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");

    read_points_bin("toy_points.bin");

    std::string shape_filename=std::string(env_p)+std::string("quad_test_ply.shp");
    test.setup_polygons(shape_filename.c_str());

    std::cout<<"running GDAL/OGR or GEOS CPU code for comparison/verification..........."<<std::endl;

    uint32_t num_print_interval=100;

    bool using_geos=true;

    //type 1: random points
    uint32_t num_pnt_samples=test.num_pnts;
    //uint32_t num_pnt_samples=200;
    test.compare_random_points(num_pnt_samples,num_print_interval,using_geos);

    //type 2: random quadrant/polygon pairs
    //uint32_t num_quad_samples=10000;
    //test.compare_matched_pairs(num_quad_samples,num_print_interval,using_geos);

    std::cout<<"h_pnt_idx_vec.size()="<<h_pnt_idx_vec.size()<<std::endl;
    thrust::copy(h_pnt_idx_vec.begin(),h_pnt_idx_vec.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    //for unknown reason, the following two lines can not be compiled in spatial_join_test_utility.cu
    //h_pnt_search_idx and h_poly_search_idx do not need to be freed as the destructor of std::vector does it
    uint32_t * h_pnt_search_idx=&(h_pnt_idx_vec[0]);
    uint32_t * h_poly_search_idx=&(h_poly_idx_vec[0]);


    bool verified=compute_mismatch(test.num_pp_pairs,test.h_org_poly_idx_vec,
        h_pnt_search_idx,test.h_pnt_len_vec,h_poly_search_idx,
        test.h_pp_pnt_idx,test.h_pp_poly_idx,
        test.h_pnt_x,test.h_pnt_y);
    std::string msg=verified ? "verified" : "mismatch";
    std::cout<<"comparison/verification result: " << msg << std::endl;
    test.tear_down();

    return(0);
}

