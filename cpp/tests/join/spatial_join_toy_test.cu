#include <random>

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

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include "spatial_join_test_utility.cuh"
#include "spatial_join_test_utility.hpp"


std::unique_ptr<column> make_numeric_column(data_type type,
                                            size_type size,
                                            mask_state state,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<column>>{});
}

template <typename T>
inline auto generate_points(std::vector<std::vector<T>> const &quads, uint32_t points_per_quad)
{
  std::vector<T> point_x(quads.size() * points_per_quad);
  std::vector<T> point_y(quads.size() * points_per_quad);

  std::seed_seq seed{time(0)};
  std::mt19937 g(seed);
  
  for (uint32_t i = 0, pos = 0; i < quads.size(); i++, pos += points_per_quad) {
    std::uniform_real_distribution<> dist_x (quads[i][0], quads[i][1]);
    std::uniform_real_distribution<> dist_y (quads[i][0], quads[i][1]);

    std::generate(point_x.begin() + pos, point_x.begin() + pos + points_per_quad, [&]() mutable {
      return dist_x(g);
    });

    std::generate(point_y.begin() + pos, point_y.begin() + pos + points_per_quad, [&]() mutable {
      return dist_y(g);
    });
  }
  return std::make_pair(std::move(point_x), std::move(point_y));
}


struct SpatialJoinNYCTaxiTest
{        
    uint32_t num_pnts=0;
    uint32_t num_quadrants=0;
    uint32_t num_pq_pairs=0;
    uint32_t num_pp_pairs=0;

    //point x/y on host
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;
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

        std::cout<<"col_poly_x"<<std::endl;
        thrust::device_ptr<const double> d_poly_xx=thrust::device_pointer_cast(col_poly_x->view().begin<double>());
        thrust::copy(d_poly_xx,d_poly_xx+col_poly_x->size(),std::ostream_iterator<const double>(std::cout, " "));std::cout<<std::endl;

        std::cout<<"col_poly_y"<<std::endl;
        thrust::device_ptr<const double> d_poly_yy=thrust::device_pointer_cast(col_poly_y->view().begin<double>());
        thrust::copy(d_poly_yy,d_poly_yy+col_poly_y->size(),std::ostream_iterator<const double>(std::cout, " "));std::cout<<std::endl;
 
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

    SBBox<double> setup_points(uint32_t const min_size)
    {
 
        std::vector<std::vector<double>> quads{{0, 2, 0, 2},
                                       {3, 4, 0, 1},
                                       {2, 3, 1, 2},
                                       {4, 6, 0, 2},
                                       {3, 4, 2, 3},
                                       {2, 3, 3, 4},
                                       {6, 7, 2, 3},
                                       {7, 8, 3, 4},
                                       {0, 4, 4, 8}};
   
        auto host_points = generate_points<double>(quads, min_size);
        this->num_pnts=std::get<0>(host_points).size();
        this->h_pnt_x        =new double[this->num_pnts];
        this->h_pnt_y        =new double[this->num_pnts];
    
        auto h_x_vec=std::get<0>(host_points);
        auto h_y_vec=std::get<1>(host_points);
        std::copy(h_x_vec.begin(),h_x_vec.end(),this->h_pnt_x );
        std::copy(h_y_vec.begin(),h_y_vec.end(),this->h_pnt_y );

        //compute the bbox of all points; outlier points may have irrational values
        //any points that do not fall within the Area of Interests (AOIs) will be assgin a special Morton code
        //AOI is user-defined and is passed to quadtree indexing and spatial join 
        double x1=*(std::min_element(h_pnt_x,h_pnt_x+this->num_pnts));
        double x2=*(std::max_element(h_pnt_x,h_pnt_x+this->num_pnts));
        double y1=*(std::min_element(h_pnt_y,h_pnt_y+this->num_pnts));
        double y2=*(std::max_element(h_pnt_y,h_pnt_y+this->num_pnts));
        std::cout<<"read_point: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;

        //create x/y columns, expose their raw pointers to be used in run_test() and populate x/y arrays
        this->col_pnt_x = make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            this->num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_pnt_x=cudf::mutable_column_device_view::create(col_pnt_x->mutable_view(), stream)->data<double>();
        assert(d_pnt_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, h_pnt_x, this->num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );

        this->col_pnt_y = make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            this->num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_pnt_y=cudf::mutable_column_device_view::create(col_pnt_y->mutable_view(), stream)->data<double>();
        assert(d_pnt_y!=nullptr);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, h_pnt_y, this->num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );
        
        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_level,uint32_t min_size)
    {
        timeval t0,t1,t2,t3,t4;

        gettimeofday(&t0, nullptr); 
        cudf::mutable_column_view pnt_x_view=col_pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=col_pnt_y->mutable_view();
        std::cout<<"run_test::num_pnts="<<col_pnt_x->size()<<std::endl;
        
        auto quadtree_pair =cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,x2,y1,y2, scale,num_level, min_size,this->mr);       
        std::unique_ptr<cudf::table> quadtree_tbl=std::move(std::get<1>(quadtree_pair));
        std::unique_ptr<cudf::column> point_indices =std::move(std::get<0>(quadtree_pair));
        this->num_quadrants=quadtree_tbl->view().num_rows();
        std::cout<<"# of quadrants="<<this->num_quadrants<<std::endl;
        gettimeofday(&t1, nullptr);
        float quadtree_time=calc_time("quadtree_tbl constrution time=",t0,t1);

        //compute polygon bbox on GPU
         auto bbox_tbl=cuspatial::polygon_bounding_boxes(col_poly_fpos->view(),col_poly_rpos->view(),
            col_poly_x->view(),col_poly_y->view(),this->mr);
            
        gettimeofday(&t2, nullptr);
        float polybbox_time=calc_time("compute polygon bbox time=",t1,t2);
        std::cout<<"# of polygon bboxes="<<bbox_tbl->view().num_rows()<<std::endl;

        //spatial filtering
        const cudf::table_view quad_view=quadtree_tbl->view();
        const cudf::table_view bbox_view=bbox_tbl->view();
      
        std::unique_ptr<cudf::table>  pq_pair_tbl = 
        	cuspatial::join_quadtree_and_bounding_boxes(quad_view, bbox_view, x1, x2, y1, y2, scale, num_level,this->mr);

        uint32_t num_pq_pairs=pq_pair_tbl->num_rows();
 
        thrust::host_vector<uint32_t> pq_poly_id(num_pq_pairs);
        thrust::host_vector<uint32_t> pq_quad_id(num_pq_pairs);
        
        HANDLE_CUDA_ERROR( cudaMemcpy(pq_poly_id.data(), pq_pair_tbl->get_column(0).view().data<uint32_t>(),num_pq_pairs* sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
        HANDLE_CUDA_ERROR( cudaMemcpy(pq_quad_id.data(), pq_pair_tbl->get_column(1).view().data<uint32_t>(),num_pq_pairs* sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
               
       printf("num_pq_pairs=%d\n",num_pq_pairs);
        for(uint32_t i=0;i<num_pq_pairs;i++)
        {
		   printf("%d, %d, %d\n",i,pq_poly_id[i],pq_quad_id[i]);
        }
            
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
                                                                     this->mr);

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
        
        const uint32_t * d_point_indices=point_indices->view().data<uint32_t>();
        this->h_point_indices=new uint32_t[this->num_pnts];
        HANDLE_CUDA_ERROR( cudaMemcpy(h_point_indices, d_point_indices,this->num_pnts * sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
                
        //setup variables for verifications
        const uint32_t *d_qt_length=quadtree_tbl->view().column(3).data<uint32_t>();
        const uint32_t *d_qt_fpos=quadtree_tbl->view().column(4).data<uint32_t>();

        this->h_qt_length=new uint32_t[this->num_quadrants];
        this->h_qt_fpos=new uint32_t[this->num_quadrants];
        assert(this->h_qt_length!=nullptr && this->h_qt_fpos!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_length, d_qt_length, this->num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_fpos, d_qt_fpos, this->num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

        this->num_pq_pairs=pq_pair_tbl->num_rows();
        const uint32_t * d_pq_poly_idx=pq_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_pq_quad_idx=pq_pair_tbl->view().column(1).data<uint32_t>();

        this->h_pq_poly_idx=new uint32_t[num_pq_pairs];
        this->h_pq_quad_idx=new uint32_t[num_pq_pairs];
        assert(this->h_pq_poly_idx!=nullptr && this->h_pq_quad_idx!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( this->h_pq_poly_idx, d_pq_poly_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( this->h_pq_quad_idx, d_pq_quad_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );

        this->num_pp_pairs=pip_pair_tbl->num_rows();
        const uint32_t *d_pp_poly_idx=pip_pair_tbl->mutable_view().column(0).data<uint32_t>();
        const uint32_t *d_pp_pnt_idx=pip_pair_tbl->mutable_view().column(1).data<uint32_t>();

        this->h_pp_poly_idx=new uint32_t[num_pp_pairs];
        this->h_pp_pnt_idx=new uint32_t[num_pp_pairs];
        assert(this->h_pp_poly_idx!=nullptr && this->h_pp_pnt_idx!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy( this->h_pp_poly_idx, d_pp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( this->h_pp_pnt_idx, d_pp_pnt_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    }
    
    void write_points_bin(const char *file_name)
    {
        CUDF_EXPECTS(file_name!=NULL,"file_name can not be NULL");
        FILE *fp=fopen(file_name,"wb");
        CUDF_EXPECTS(fp!=NULL, "can not open file for output");
        
        uint32_t tb=0;
        CUDF_EXPECTS(fwrite(&(this->num_pnts),sizeof(uint32_t),1,fp)==1,"writting num_pnt failed");
        CUDF_EXPECTS(fwrite(&(this->num_quadrants),sizeof(uint32_t),1,fp)==1,"writting num_quadrants failed");
        CUDF_EXPECTS(fwrite(&(this->num_pq_pairs),sizeof(uint32_t),1,fp)==1,"writting num_pq_pairs failed");
        CUDF_EXPECTS(fwrite(&(this->num_pp_pairs),sizeof(uint32_t),1,fp)==1,"writting num_pp_pairs failed");
        tb+=4*sizeof(uint32_t);
        
        CUDF_EXPECTS(fwrite(this->h_pnt_x,sizeof(double),this->num_pnts,fp)==this->num_pnts,"writting h_pnt_x failed");
        CUDF_EXPECTS(fwrite(this->h_pnt_y,sizeof(double),this->num_pnts,fp)==this->num_pnts,"writting h_pnt_y failed");
        CUDF_EXPECTS(fwrite(this->h_point_indices,sizeof(uint32_t),this->num_pnts,fp)==this->num_pnts,"writting h_point_indices failed");
        tb+=(this->num_pnts*(sizeof(double)*2+sizeof(uint32_t)));
        thrust::copy(this->h_pnt_x,this->h_pnt_x+this->num_pnts,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;    
        thrust::copy(this->h_pnt_y,this->h_pnt_y+this->num_pnts,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;   
        thrust::copy(this->h_point_indices,this->h_point_indices+this->num_pnts,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;   
        
        CUDF_EXPECTS(fwrite(this->h_qt_length,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"writting h_qt_length failed");
        CUDF_EXPECTS(fwrite(this->h_qt_fpos,sizeof(uint32_t),this->num_quadrants,fp)==this->num_quadrants,"writting h_qt_fpos failed");
        tb+=(this->num_quadrants*sizeof(uint32_t)*2);
        thrust::copy(this->h_qt_length,this->h_qt_length+this->num_quadrants,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;    
        thrust::copy(this->h_qt_fpos,this->h_qt_fpos+this->num_quadrants,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;   
        
        CUDF_EXPECTS(fwrite(this->h_pq_quad_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"writting h_pq_quad_idx failed");
        CUDF_EXPECTS(fwrite(this->h_pq_poly_idx,sizeof(uint32_t),this->num_pq_pairs,fp)==this->num_pq_pairs,"writting h_pq_poly_idx failed");
        tb+=(this->num_pq_pairs*sizeof(uint32_t)*2);
        thrust::copy(this->h_pq_quad_idx,this->h_pq_quad_idx+this->num_pq_pairs,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;    
        thrust::copy(this->h_pq_poly_idx,this->h_pq_poly_idx+this->num_pq_pairs,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;   
        
        
        CUDF_EXPECTS(fwrite(this->h_pp_poly_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"writting h_pp_poly_idx failed");
        CUDF_EXPECTS(fwrite(this->h_pp_pnt_idx,sizeof(uint32_t),this->num_pp_pairs,fp)==this->num_pp_pairs,"writting h_pp_pnt_idx failed");
        tb+=(this->num_pp_pairs*sizeof(uint32_t)*2);
        thrust::copy(this->h_pp_poly_idx,this->h_pp_poly_idx+this->num_pp_pairs,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;    
        thrust::copy(this->h_pp_pnt_idx,this->h_pp_pnt_idx+this->num_pp_pairs,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;   

        for(uint32_t i=0;i<this->num_pp_pairs;i++)
        {
           uint32_t pid=this->h_point_indices[this->h_pp_pnt_idx[i]];
           printf("%d,%10.5f, %10.5f, %d\n",i,this->h_pnt_x[pid],this->h_pnt_y[pid],this->h_pp_poly_idx[i]);
	}
  
        printf("total bytes=%d\n",tb);  
    }    
};


int main()
{
    SpatialJoinNYCTaxiTest test;
    
    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");
    
    std::cout<<"loading point data..........."<<std::endl;
    uint32_t const max_depth{3};
    uint32_t const min_size{400};
    double const scale{1.0};
    test.setup_points(min_size);        

    std::cout<<"loading polygon data..........."<<std::endl;
    std::string shape_filename=std::string(env_p)+std::string("quad_test_ply.shp");     
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;
    SBBox<double> aoi=test.setup_polygons(shape_filename.c_str());

    //verify all polygon vertices (x,y) in the shapefile are between [0.0,8.0) and [0.0,8.0) 
    
    double poly_x1=thrust::get<0>(aoi.first);
    double poly_y1=thrust::get<1>(aoi.first);
    double poly_x2=thrust::get<0>(aoi.second);
    double poly_y2=thrust::get<1>(aoi.second);
    
    printf("x1=%10.5f y1=%10.5f x2=%10.5f y2=%10.5f\n",poly_x1,poly_y1,poly_x2,poly_y2);
    
    std::cout<<"running test on toy data..........."<<std::endl;
    test.run_test(0.0,0.0,8.0,8.0,scale,max_depth,min_size);
    
    test.write_points_bin("toy_points.bin");
    
    return(0);
}

