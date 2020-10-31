
#include <cudf_test/base_fixture.hpp>

#include "spatial_join_test_utility.hpp"
#include "spatial_join_geo_utility.hpp"

struct SpatialJoinNYCTaxiVerify : public cudf::test::BaseFixture
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

        this->h_ogr_polygon_vec.clear();
        this->h_geos_polygon_vec.clear();
        this->h_org_poly_idx_vec.clear();

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
        this->num_poly=f_len_v.size();
        this->num_ring=r_len_v.size();
        this->num_vertex=x_v.size();

        uint32_t *h_poly_flen=new uint32_t[num_poly];
        uint32_t *h_poly_rlen=new uint32_t[num_ring];
        assert(h_poly_flen!=nullptr && h_poly_rlen!=nullptr);

        this->h_poly_x=new double [num_vertex];
        this->h_poly_y=new double [num_vertex];
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
        std::cout<<"compare_random_points: num_quadrants="<<this->num_quadrants
            <<" num_pp_pair="<<this->num_pp_pairs<<" num_samples="<<num_samples<<std::endl;

        std::vector<uint32_t> rand_indices;
        gen_rand_idx(rand_indices,this->num_pnts,num_samples);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);

        //h_pnt_idx_vec, h_pnt_len_vec and h_poly_idx_vec will be cleared first

        if(using_geos)
        {
            rand_points_geos_pip_test(num_print_interval,rand_indices, this->h_geos_polygon_vec,this->h_pnt_idx_vec,
                this->h_pnt_len_vec,this->h_poly_idx_vec,this->h_pnt_x,this->h_pnt_y,this->h_point_indices);
        }
        else
        {
            rand_points_ogr_pip_test(num_print_interval,rand_indices, this->h_ogr_polygon_vec,this->h_pnt_idx_vec,
                this->h_pnt_len_vec,this->h_poly_idx_vec,this->h_pnt_x,this->h_pnt_y,this->h_point_indices);
         }
        gettimeofday(&t1, nullptr);
        float cpu_time=calc_time("cpu random sampling computing time = ",t0,t1);
    }

    void compare_matched_pairs(uint32_t num_samples,uint32_t num_print_interval,bool using_geos)
    {
        std::cout<<"compare_matched_pairs: num_quadrants="<<this->num_quadrants<<" num_pq_pairs"<<this->num_pq_pairs
            <<" num_pp_pair="<<this->num_pp_pairs<<" num_samples="<<num_samples<<std::endl;

        std::vector<uint32_t> rand_indices;
        gen_rand_idx(rand_indices,this->num_pq_pairs,num_samples);

        timeval t0,t1;
        gettimeofday(&t0, nullptr);

        if(using_geos)
        {
            matched_pairs_geos_pip_test(num_print_interval,rand_indices,
                this->h_pq_quad_idx,this->h_pq_poly_idx,this->h_qt_length,this->h_qt_fpos,
                this->h_geos_polygon_vec,this->h_pnt_idx_vec,this->h_pnt_len_vec,this->h_poly_idx_vec,
                this->h_pnt_x,this->h_pnt_y,this->h_point_indices);
        }
        else
        {
            matched_pairs_ogr_pip_test(num_print_interval,rand_indices,
                this->h_pq_quad_idx,this->h_pq_poly_idx,this->h_qt_length,this->h_qt_fpos,
                this->h_ogr_polygon_vec,this->h_pnt_idx_vec,this->h_pnt_len_vec,this->h_poly_idx_vec,
                this->h_pnt_x,this->h_pnt_y,this->h_point_indices);

        }
        gettimeofday(&t1, nullptr);
        float cpu_time=calc_time("cpu matched-pair computing time",t0,t1);
    }

    void read_nyc_taxi(const char *file_name)
    {
        CUDF_EXPECTS(file_name!=NULL,"file_name can not be NULL");
        FILE *fp=fopen(file_name,"rb");
        CUDF_EXPECTS(fp!=NULL, "can not open file for input");
        CUDF_EXPECTS(fread(&(this->num_pnts),sizeof(uint32_t),1,fp)==1,"reading num_pnt failed");
        CUDF_EXPECTS(fread(&(this->num_quadrants),sizeof(uint32_t),1,fp)==1,"reading num_quadrants failed");
        CUDF_EXPECTS(fread(&(this->num_pq_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pq_pairs failed");
        CUDF_EXPECTS(fread(&(this->num_pp_pairs),sizeof(uint32_t),1,fp)==1,"reading num_pp_pairs failed");
        std::cout<<"num_pnts="<<this->num_pnts<<" num_quadrants="<<this->num_quadrants<<" num_pq_pairs="<<this->num_pq_pairs<<" num_pp_pairs="<<this->num_pp_pairs<<std::endl;

        std::cout<<"reading points..."<<std::endl;
        this->h_pnt_x=new double[this->num_pnts];
        this->h_pnt_y=new double[this->num_pnts];
        this->h_point_indices= new uint32_t[this->num_pnts];
        CUDF_EXPECTS( this->h_pnt_x!=NULL && this->h_pnt_y!=NULL && this->h_point_indices!=NULL	 ,"allocating memory for points on host failed");

        CUDF_EXPECTS(fread(this->h_pnt_x,sizeof(double),this->num_pnts,fp)==this->num_pnts,"reading h_pnt_x failed");
        CUDF_EXPECTS(fread(this->h_pnt_y,sizeof(double),this->num_pnts,fp)==this->num_pnts,"reading h_pnt_y failed");
        CUDF_EXPECTS(fread(this->h_point_indices,sizeof(uint32_t),this->num_pnts,fp)==this->num_pnts,"reading h_point_indices failed");

        std::cout<<"reading quadrants..."<<std::endl;
        this->h_qt_length=new uint32_t[this->num_quadrants];
        this->h_qt_fpos=new uint32_t[this->num_quadrants];
        CUDF_EXPECTS( this->h_qt_length!=NULL && this->h_qt_fpos!=NULL,"allocating memory for quadrants on host failed");

        CUDF_EXPECTS(fread(this->h_qt_length,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"reading h_qt_length failed");
        CUDF_EXPECTS(fread(this->h_qt_fpos,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"reading h_qt_fpos failed");

        std::cout<<"reading quadrant/polygon pairs..."<<std::endl;
        this->h_pq_quad_idx=new uint32_t[this->num_pq_pairs];
        this->h_pq_poly_idx=new uint32_t[this->num_pq_pairs];
        CUDF_EXPECTS( this->h_pq_poly_idx!=NULL && this->h_pq_quad_idx!=NULL,"allocating memory for quadrant-polygon pairs on host failed");

        CUDF_EXPECTS(fread(this->h_pq_quad_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"reading h_pq_quad_idx failed");
        CUDF_EXPECTS(fread(this->h_pq_poly_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"reading h_pq_poly_idx failed");

        std::cout<<"reading point/polygon pairs..."<<std::endl;
        this->h_pp_poly_idx=new uint32_t[this->num_pp_pairs];
        this->h_pp_pnt_idx=new uint32_t[this->num_pp_pairs];
        CUDF_EXPECTS(this->h_pp_poly_idx!=NULL && this->h_pp_pnt_idx!=NULL,"allocating memory for point-polygon pairs on host failed");

        CUDF_EXPECTS(fread(this->h_pp_poly_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"reading h_pp_poly_idx failed");
        CUDF_EXPECTS(fread(this->h_pp_pnt_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"reading h_pp_pnt_idx failed");

    }

    void tear_down()
    {
        delete[] this->h_poly_x; this->h_poly_x=nullptr;
        delete[] this->h_poly_y; this->h_poly_y=nullptr;

        delete[] this->h_pnt_x; this->h_pnt_x=nullptr;
        delete[] h_pnt_y; h_pnt_y=nullptr;

        delete[] this->h_pq_quad_idx; this->h_pq_quad_idx=nullptr;
        delete[] h_pq_poly_idx; h_pq_poly_idx=nullptr;

        delete[] this->h_qt_length; this->h_qt_length=nullptr;
        delete[] this->h_qt_fpos; this->h_qt_fpos=nullptr;
    }

};

/*
 * There could be multple configureations (minior ones are inside parentheses):
 * pick one of three polygon datasets
 * choose from compare_random_points and compare_matched_pairs
*/

TEST_F(SpatialJoinNYCTaxiVerify, verify)
{
    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");

    //#0: NYC taxi zone: 263 polygons
    //from https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip
    //#1: NYC Community Districts: 71 polygons
    //from https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_11aav.zip
    //#2: NYC Census Tract 2000 data: 2216 polygons
    //from: https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2000_11aav.zip

    //note that the polygons and the points need to use the same projection
    //all the three polygon datasets use epsg:2263 (unit is foot) for NYC/Long Island area

    enum POLYID {taxizone_id=0,cd_id,ct_id};
    POLYID sel_id=taxizone_id;

    const char * shape_files[]={"taxi_zones.shp","nycd_11a_av/nycd.shp","nyct2000_11a_av/nyct2000.shp"};

    const char * bin_files[]={"nyc_taxizone_2009_1.bin","nyc_cd_2009_12.bin","nyc_ct_2009_12.bin"};

    read_nyc_taxi(bin_files[sel_id]);

    std::cout<<"loading NYC polygon data..........."<<std::endl;

    std::string shape_filename=std::string(env_p)+std::string(shape_files[sel_id]);

    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    //uint8_t poly_type=2; //multi-polygons only
    //uint8_t poly_type=1; //single-polygons only
    uint8_t poly_type=0; //all polygons

    this->setup_polygons(shape_filename.c_str());

    std::cout<<"running GDAL/OGR or GEOS CPU code for comparison/verification..........."<<std::endl;

    uint32_t num_print_interval=100;

    bool using_geos=true;

    //type 1: random points
    //uint32_t num_pnt_samples=this->num_pnts;
    uint32_t num_pnt_samples=10000;
    this->compare_random_points(num_pnt_samples,num_print_interval,using_geos);

    //type 2: random quadrant/polygon pairs
    //uint32_t num_quad_samples=10000;
    //this->compare_matched_pairs(num_quad_samples,num_print_interval,using_geos);

    //for unknown reason, the following two lines can not be compiled in spatial_join_test_utility.cu
    //h_pnt_search_idx and h_poly_search_idx do not need to be freed as the destructor of std::vector does it
    uint32_t * h_pnt_search_idx=&(h_pnt_idx_vec[0]);
    uint32_t * h_poly_search_idx=&(h_poly_idx_vec[0]);

    bool verified=compute_mismatch(this->num_pp_pairs,this->h_org_poly_idx_vec,
        h_pnt_search_idx,this->h_pnt_len_vec,h_poly_search_idx,
        this->h_pp_pnt_idx,this->h_pp_poly_idx,
        this->h_pnt_x,this->h_pnt_y);
    std::string msg=verified ? "verified" : "mismatch";
    std::cout<<"comparison/verification result: " << msg << std::endl;
    this->tear_down();

}//TEST_F

