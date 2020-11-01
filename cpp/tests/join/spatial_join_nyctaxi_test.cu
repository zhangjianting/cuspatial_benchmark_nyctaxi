
#include <cuspatial/error.hpp>
#include <cuspatial/shapefile_reader.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/polygon_bounding_box.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>
#include "spatial_join_test_utility.cuh"
#include "spatial_join_test_utility.hpp"

std::unique_ptr<cudf::column> make_numeric_column(cudf::data_type type,
                                            cudf::size_type size,
                                            cudf::mask_state state,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  //CUDF_FUNC_RANGE();
  //CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return std::make_unique<cudf::column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<cudf::column>>{});
}

struct SpatialJoinNYCTaxiTest
{        
    uint32_t num_pnts=0;
    uint32_t num_quadrants=0;
    uint32_t num_pq_pairs=0;
    uint32_t num_pp_pairs=0;

    //point x/y on host
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;
    double *d_pnt_x=NULL,*d_pnt_y=NULL;
    uint32_t *h_point_indices=nullptr;

    //quadtree length/fpos
    uint32_t *h_qt_length=nullptr,*h_qt_fpos=nullptr;   

    //quadrant/polygon pairs
    uint32_t *h_pq_quad_idx=nullptr,*h_pq_poly_idx=nullptr;   
    
    //point/polygon pairs on device; shared between run_test and compute_mismatch
    //the life span of d_pp_pnt_idx/d_pp_poly_idx depends on pip_pair_tbl
    uint32_t *h_pp_pnt_idx=nullptr,*h_pp_poly_idx=nullptr;
    std::unique_ptr<cudf::column> col_pnt_x,col_pnt_y;
    std::unique_ptr<cudf::column> col_poly_fpos,col_poly_rpos,col_poly_x,col_poly_y;    
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_current_device_resource();

    SBBox<double> setup_polygons(const char *file_name)
    {
        std::vector<std::unique_ptr<cudf::column>> polygon_columns = cuspatial::read_polygon_shapefile(file_name);
        std::cout<<"setup_polygons::polygon_columns="<<polygon_columns.size()<<std::endl;
        
        col_poly_fpos=std::move(polygon_columns.at(0));
        col_poly_rpos=std::move(polygon_columns.at(1));
        col_poly_x=std::move(polygon_columns.at(2));
        col_poly_y=std::move(polygon_columns.at(3));
        std::cout<<"setup_polygons::col_poly_fpos.size="<<col_poly_fpos->size()<<" col_poly_rpos.size="<<col_poly_rpos->size()<<std::endl;
        std::cout<<"setup_polygons::x.size="<<col_poly_x->size()<<" y.size="<<col_poly_y->size()<<std::endl;
        
        std::cout<<"col_poly_fpos"<<std::endl;
        thrust::device_ptr<const uint32_t> d_poly_fpos=thrust::device_pointer_cast(col_poly_fpos->view().begin<uint32_t>());
        thrust::copy(d_poly_fpos,d_poly_fpos+col_poly_fpos->size(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl; 
  
        std::cout<<"col_poly_rpos"<<std::endl;
        thrust::device_ptr<const uint32_t> d_poly_rpos=thrust::device_pointer_cast(col_poly_rpos->view().begin<uint32_t>());
        thrust::copy(d_poly_rpos,d_poly_rpos+col_poly_rpos->size(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl; 
               
        /*const double *x1_p=thrust::min_element(thrust::device,col_poly_x.begin<double>(),col_poly_x.end<double>());
        const double *x2_p=thrust::max_element(thrust::device,col_poly_x.begin<double>(),col_poly_x.end<double>());
        const double *y1_p=thrust::min_element(thrust::device,col_poly_y.begin<double>(),col_poly_y.end<double>());
        const double *y2_p=thrust::max_element(thrust::device,col_poly_y.begin<double>(),col_poly_y.end<double>());
        double x1,y1,x2,y2;
        cudaMemcpy(&x1,x1_p, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&y1,y1_p, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&x2,x2_p, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&y2,y2_p, sizeof(double), cudaMemcpyDeviceToHost);*/ 
        
        std::unique_ptr<cudf::scalar> x1_s=cudf::reduce(col_poly_x->view(), cudf::make_min_aggregation(),cudf::data_type{cudf::type_id::FLOAT64});
        std::unique_ptr<cudf::scalar> x2_s=cudf::reduce(col_poly_x->view(), cudf::make_max_aggregation(),cudf::data_type{cudf::type_id::FLOAT64});
        std::unique_ptr<cudf::scalar> y1_s=cudf::reduce(col_poly_y->view(), cudf::make_min_aggregation(),cudf::data_type{cudf::type_id::FLOAT64});
        std::unique_ptr<cudf::scalar> y2_s=cudf::reduce(col_poly_y->view(), cudf::make_max_aggregation(),cudf::data_type{cudf::type_id::FLOAT64});
        
        // auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

        auto x1=static_cast<cudf::scalar_type_t<double>*>(x1_s.get())->value();
        auto y1=static_cast<cudf::scalar_type_t<double>*>(y1_s.get())->value();
        auto x2=static_cast<cudf::scalar_type_t<double>*>(x2_s.get())->value();
        auto y2=static_cast<cudf::scalar_type_t<double>*>(y2_s.get())->value();
  
        std::cout<<"x1="<<x1<<" x2="<<x2<<" y1="<<y1<<" y2="<<y2<<std::endl;
        
        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    SBBox<double> setup_points(const char * file_name, uint32_t first_n)
    {
        num_pnts=0;
        //read invidual data file  
        std::vector<uint32_t> len_vec;
        std::vector<double *> x_vec;
        std::vector<double *> y_vec;
        uint32_t num=0;
        
        FILE *fp=nullptr;
        if((fp=fopen(file_name,"r"))==nullptr)
        {
           std::cout<<"Failed to open point catalog file "<<file_name<<std::endl;
           exit(-2);          
        }
        while(!feof(fp))
        {
             char str[500];
             int n1=fscanf(fp,"%s",str);
             std::cout<<"processing point data file "<<str<<std::endl;
             double *tmp_x=nullptr,*tmp_y=nullptr;
             size_t temp_len=read_point_binary(str,tmp_x,tmp_y);
             assert(tmp_x!=nullptr && tmp_y!=nullptr);
             num++;
             len_vec.push_back(temp_len);
             x_vec.push_back(tmp_x);
             y_vec.push_back(tmp_y);
             if(first_n>0 && num>=first_n) break;
        }    
        fclose(fp);

        //prepare memory allocation
        for(uint32_t i=0;i<num;i++)
            num_pnts+=len_vec[i];
        uint32_t p=0;
        h_pnt_x=new double[num_pnts];
        h_pnt_y=new double[num_pnts];
        assert(h_pnt_x!=nullptr && h_pnt_y!=nullptr);
        
        //concatination
        for(uint32_t i=0;i<num;i++)
        {
            double *tmp_x=x_vec[i];
            double *tmp_y=y_vec[i];
            assert(tmp_x!=nullptr && tmp_y!=nullptr);
            int len=len_vec[i];
            std::copy(tmp_x,tmp_x+len,h_pnt_x+p);
            std::copy(tmp_y,tmp_y+len,h_pnt_y+p);
            p+=len;
            delete[] tmp_x;
            delete[] tmp_y;
        }
        assert(p==num_pnts);

        //compute the bbox of all points; outlier points may have irrational values
        //any points that do not fall within the Area of Interests (AOIs) will be assgin a special Morton code
        //AOI is user-defined and is passed to quadtree indexing and spatial join 
        double x1=*(std::min_element(h_pnt_x,h_pnt_x+num_pnts));
        double x2=*(std::max_element(h_pnt_x,h_pnt_x+num_pnts));
        double y1=*(std::min_element(h_pnt_y,h_pnt_y+num_pnts));
        double y2=*(std::max_element(h_pnt_y,h_pnt_y+num_pnts));
        std::cout<<"read_point_catalog: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;

        //create x/y columns, expose their raw pointers to be used in run_test() and populate x/y arrays
        col_pnt_x = make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_x=cudf::mutable_column_device_view::create(col_pnt_x->mutable_view(), stream)->data<double>();
        assert(d_pnt_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, h_pnt_x, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );

        col_pnt_y = make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_y=cudf::mutable_column_device_view::create(col_pnt_y->mutable_view(), stream)->data<double>();
        assert(d_pnt_y!=nullptr);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, h_pnt_y, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );
        
        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_level,uint32_t min_size)
    {
        timeval t0,t1,t2,t3,t4;

        gettimeofday(&t0, nullptr); 
        cudf::mutable_column_view pnt_x_view=col_pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=col_pnt_y->mutable_view();
        std::cout<<"run_test::num_pnts="<<col_pnt_x->size()<<std::endl;
        
        auto quadtree_pair =cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,x2,y1,y2, scale,num_level, min_size,mr);       
        std::unique_ptr<cudf::table> quadtree_tbl=std::move(std::get<1>(quadtree_pair));
        std::unique_ptr<cudf::column> point_indices =std::move(std::get<0>(quadtree_pair));
        num_quadrants=quadtree_tbl->view().num_rows();
        std::cout<<"# of quadrants="<<num_quadrants<<std::endl;
        gettimeofday(&t1, nullptr);
        float quadtree_time=calc_time("quadtree_tbl constrution time=",t0,t1);

        //compute polygon bbox on GPU
         auto bbox_tbl=cuspatial::polygon_bounding_boxes(col_poly_fpos->view(),col_poly_rpos->view(),
            col_poly_x->view(),col_poly_y->view(),mr);
            
        gettimeofday(&t2, nullptr);
        float polybbox_time=calc_time("compute polygon bbox time=",t1,t2);
        std::cout<<"# of polygon bboxes="<<bbox_tbl->view().num_rows()<<std::endl;

        //spatial filtering
        const cudf::table_view quad_view=quadtree_tbl->view();
        const cudf::table_view bbox_view=bbox_tbl->view();
      
        std::unique_ptr<cudf::table>  pq_pair_tbl = 
        	cuspatial::join_quadtree_and_bounding_boxes(quad_view, bbox_view, x1, x2, y1, y2, scale, num_level,mr);

            
        gettimeofday(&t3, nullptr);
        float filtering_time=calc_time("spatial filtering time=",t2,t3);
        std::cout<<"# of polygon/quad pairs="<<pq_pair_tbl->view().num_rows()<<std::endl;

        //spatial refinement 
        std::unique_ptr<cudf::table> pip_pair_tbl = cuspatial::quadtree_point_in_polygon(pq_pair_tbl->view(),
                                                                     quadtree_tbl->view(),
                                                                     point_indices->view(),
                                                                     col_pnt_x->view(),
                                                                     col_pnt_y->view(),
                                                                     col_poly_fpos->view(),
                                                                     col_poly_rpos->view(),
                                                                     col_poly_x->view(),
                                                                     col_poly_y->view(),
                                                                     mr);
        gettimeofday(&t4, nullptr);
        float refinement_time=calc_time("spatial refinement time=",t3,t4);
        std::cout<<"# of polygon/point pairs="<<pip_pair_tbl->view().num_rows()<<std::endl;

        gettimeofday(&t1, nullptr);
        float gpu_time=calc_time("gpu end-to-end computing time=",t0,t1);

        //summierize runtimes
        float  runtimes[4]={quadtree_time,polybbox_time,filtering_time,refinement_time};
        const char  *msg_type[4]={"quadtree_time","polybbox_time","filtering_time","refinement_time"};
        float total_time=0;
        for(uint32_t i=0;i<4;i++)
        {
            std::cout<<msg_type[i]<<"= "<<runtimes[i]<<std::endl;
            total_time+=runtimes[i];
        }
        std::cout<<std::endl;
        std::cout<<"total_time="<<total_time<<std::endl;
        std::cout<<"gpu end-to-tend time"<<gpu_time<<std::endl;
        
        //copy back sorted points to CPU for verification
        //HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_x, d_pnt_x,num_pnts * sizeof(double), cudaMemcpyDeviceToHost ) );
        //HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_y, d_pnt_y,num_pnts * sizeof(double), cudaMemcpyDeviceToHost ) );

        const uint32_t * d_point_indices=point_indices->view().data<uint32_t>();
        h_point_indices=new uint32_t[num_pnts];
        HANDLE_CUDA_ERROR( cudaMemcpy(h_point_indices, d_point_indices,num_pnts * sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
        
        //setup variables for verifications
        const uint32_t *d_qt_length=quadtree_tbl->view().column(3).data<uint32_t>();
        const uint32_t *d_qt_fpos=quadtree_tbl->view().column(4).data<uint32_t>();

        h_qt_length=new uint32_t[num_quadrants];
        h_qt_fpos=new uint32_t[num_quadrants];
        assert(h_qt_length!=nullptr && h_qt_fpos!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_length, d_qt_length, num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_fpos, d_qt_fpos, num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

        num_pq_pairs=pq_pair_tbl->num_rows();
        const uint32_t * d_pq_poly_idx=pq_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_pq_quad_idx=pq_pair_tbl->view().column(1).data<uint32_t>();

        h_pq_poly_idx=new uint32_t[num_pq_pairs];
        h_pq_quad_idx=new uint32_t[num_pq_pairs];
        assert(h_pq_poly_idx!=nullptr && h_pq_quad_idx!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( h_pq_poly_idx, d_pq_poly_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( h_pq_quad_idx, d_pq_quad_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

        num_pp_pairs=pip_pair_tbl->num_rows();
        const uint32_t *d_pp_poly_idx=pip_pair_tbl->mutable_view().column(0).data<uint32_t>();
        const uint32_t *d_pp_pnt_idx=pip_pair_tbl->mutable_view().column(1).data<uint32_t>();

        h_pp_poly_idx=new uint32_t[num_pp_pairs];
        h_pp_pnt_idx=new uint32_t[num_pp_pairs];
        assert(h_pp_poly_idx!=nullptr && h_pp_pnt_idx!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_poly_idx, d_pp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_pnt_idx, d_pp_pnt_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    }
    
    void write_nyc_taxi(const char *file_name)
    {
        CUDF_EXPECTS(file_name!=NULL,"file_name can not be NULL");
        FILE *fp=fopen(file_name,"wb");
        CUDF_EXPECTS(fp!=NULL, "can not open file for output");
        
        CUDF_EXPECTS(fwrite(&(num_pnts),sizeof(uint32_t),1,fp)==1,"writting num_pnt failed");
        CUDF_EXPECTS(fwrite(&(num_quadrants),sizeof(uint32_t),1,fp)==1,"writting num_quadrants failed");
        CUDF_EXPECTS(fwrite(&(num_pq_pairs),sizeof(uint32_t),1,fp)==1,"writting num_pq_pairs failed");
        CUDF_EXPECTS(fwrite(&(num_pp_pairs),sizeof(uint32_t),1,fp)==1,"writting num_pp_pairs failed");
        
        CUDF_EXPECTS(fwrite(h_pnt_x,sizeof(double),num_pnts,fp)==num_pnts,"writting h_pnt_x failed");
        CUDF_EXPECTS(fwrite(h_pnt_y,sizeof(double),num_pnts,fp)==num_pnts,"writting h_pnt_y failed");
        CUDF_EXPECTS(fwrite(h_point_indices,sizeof(uint32_t),num_pnts,fp)==num_pnts,"writting h_point_indices failed");
        
        CUDF_EXPECTS(fwrite(h_qt_length,sizeof(uint32_t),num_quadrants,fp)==num_quadrants,"writting h_qt_length failed");
        CUDF_EXPECTS(fwrite(h_qt_fpos,sizeof(uint32_t),num_quadrants,fp)==num_quadrants,"writting h_qt_fpos failed");
        
        CUDF_EXPECTS(fwrite(h_pq_quad_idx,sizeof(uint32_t),num_pq_pairs,fp)==num_pq_pairs,"writting h_pq_quad_idx failed");
        CUDF_EXPECTS(fwrite(h_pq_poly_idx,sizeof(uint32_t),num_pq_pairs,fp)==num_pq_pairs,"writting h_pq_poly_idx failed");
        
        CUDF_EXPECTS(fwrite(h_pp_poly_idx,sizeof(uint32_t),num_pp_pairs,fp)==num_pp_pairs,"writting h_pp_poly_idx failed");
        CUDF_EXPECTS(fwrite(h_pp_pnt_idx,sizeof(uint32_t),num_pp_pairs,fp)==num_pp_pairs,"writting h_pp_pnt_idx failed");
    }    
};


int main()
{
    SpatialJoinNYCTaxiTest test;
    
    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");
    
    const uint32_t num_level=15;
    const uint32_t min_size=512;
    const uint32_t first_n=12; 

    std::cout<<"loading NYC taxi pickup locations..........."<<std::endl;
    
    //from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page; 
    //pickup/drop-off locations are extracted and the lon/lat coordiates are converted to epsg:2263 projection
        
    //a catalog file is simply a collection of invidual binary data files with a pre-defined structure
    //each line repersents a data file, e.g., pickup+drop-off locations for a month
    std::string catalog_filename=std::string(env_p)+std::string("2009.cat"); 
    std::cout<<"Using catalog file "<<catalog_filename<<std::endl;
    test.setup_points(catalog_filename.c_str(),first_n);

    std::cout<<"loading NYC polygon data..........."<<std::endl;

    enum POLYID {taxizone_id=0,cd_id,ct_id};    
    POLYID sel_id=taxizone_id;

    const char * shape_files[]={"taxi_zones.shp","nycd_11a_av/nycd.shp","nyct2000_11a_av/nyct2000.shp"};
    
    const char * bin_files[]={"nyc_taxizone_2009_1.bin","nyc_cd_2009_12.bin","nyc_ct_2009_12.bin"};
 
    std::cout<<"loading NYC polygon data..........."<<std::endl;

    std::string shape_filename=std::string(env_p)+std::string(shape_files[sel_id]); 
    
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    SBBox<double> aoi=test.setup_polygons(shape_filename.c_str());

    double poly_x1=thrust::get<0>(aoi.first);
    double poly_y1=thrust::get<1>(aoi.first);
    double poly_x2=thrust::get<0>(aoi.second);
    double poly_y2=thrust::get<1>(aoi.second);
    
    double width=poly_x2-poly_x1;
    double height=poly_y2-poly_y1;
    double length=(width>height)?width:height;
    double scale=length/((1<<num_level)+2);
    double bbox_x1=poly_x1-scale;
    double bbox_y1=poly_y1-scale;
    double bbox_x2=poly_x2+scale; 
    double bbox_y2=poly_y2+scale;
    printf("Area of Interests: length=%15.10f scale=%15.10f\n",length,scale);

    std::cout<<"running test on NYC taxi trip data..........."<<std::endl;
    test.run_test(bbox_x1,bbox_y1,bbox_x2,bbox_y2,scale,num_level,min_size);
    
    write_nyc_taxi(bin_files[sel_id]);
    
    return(0); 
}

