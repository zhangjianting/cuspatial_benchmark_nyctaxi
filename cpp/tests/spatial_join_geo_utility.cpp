#include <time.h>
#include <sys/time.h>
#include <random>
#include <cassert>
#include <iostream>
#include <ogrsf_frmts.h>
#include <geos_c.h>

#include "spatial_join_test_utility.hpp"
#include "spatial_join_geo_utility.hpp"

//placeholder for structus and functions needed to run NYC taxi experiments
//they will be incorporated into the new io module of cuspatial in a later release


void VertexFromLinearRing(OGRLinearRing const& poRing, std::vector<double> &aPointX,
    std::vector<double> &aPointY,std::vector<int> &aPartSize )
{
    int nCount = poRing.getNumPoints();
    int nNewCount = aPointX.size() + nCount;

    aPointX.reserve( nNewCount );
    aPointY.reserve( nNewCount );
    for (int i = nCount - 1; i >= 0; i-- )
    {
        aPointX.push_back( poRing.getX(i));
        aPointY.push_back( poRing.getY(i));
    }
    aPartSize.push_back( nCount );
}

/*
    *Read a Polygon (could be with multiple rings) into x/y/size vectors
*/
void LinearRingFromPolygon(OGRPolygon const & poPolygon, std::vector<double> &aPointX,
    std::vector<double> &aPointY,std::vector<int> &aPartSize )
{

    VertexFromLinearRing( *(poPolygon.getExteriorRing()),aPointX, aPointY, aPartSize );
    for(int i = 0; i < poPolygon.getNumInteriorRings(); i++ )
        VertexFromLinearRing( *(poPolygon.getInteriorRing(i)),aPointX, aPointY, aPartSize );
}

/*
 * Read a Geometry (could be MultiPolygon/GeometryCollection) into x/y/size vectors
*/

void PolygonFromGeometry(OGRGeometry const *poShape, std::vector<double> &aPointX,
    std::vector<double> &aPointY,std::vector<int> &aPartSize )
{
    OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());

    if (eFlatType == wkbMultiPolygon || eFlatType == wkbGeometryCollection )
    {
        printf("warning: wkbMultiPolygon or wkbGeometryCollection..................\n");
        OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;
        for(int i = 0; i < poGC->getNumGeometries(); i++ )
        {
            OGRGeometry *poGeom=poGC->getGeometryRef(i);
            PolygonFromGeometry(poGeom,aPointX, aPointY, aPartSize );
        }
    }
    else if (eFlatType == wkbPolygon)
        LinearRingFromPolygon(*((OGRPolygon *) poShape),aPointX, aPointY, aPartSize );
    else
    {
       printf("error: must be polygonal geometry.\n" );
       exit(-1);
    }
}

int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
     std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v,
         uint8_t type, std::vector<OGRGeometry *>& polygon_vec, std::vector<uint32_t>& idx_vec)
{
    uint32_t num_feature=0,num_seq=0;
    OGR_L_ResetReading(layer );
    OGRFeatureH hFeat;

    while( (hFeat = OGR_L_GetNextFeature( layer )) != nullptr )
    {
        OGRGeometry *poShape=(OGRGeometry *)OGR_F_GetGeometryRef( hFeat );

        if(poShape==nullptr)
        {
            printf("Invalid Shape\n");
            exit(-1);
        }
        if(type!=0&&type!=1&&type!=2)
        {
            printf("unknown type to setup polygons (0 for all, 1 for simple polygon and 2 for multi-polygon): type=%d\n",type);
            exit(-1);
        }

        if(type==1 ||type==2)
        {

//highlight logic for type 1/2 by increasing their indentations to top level
if(type==1)
{
    if(poShape->getGeometryType()!=wkbPolygon) //wkbGeometryType=3
    {
       OGR_F_Destroy( hFeat );
       num_seq++;
       continue;
    }
}
if(type==2)
{
    if(poShape->getGeometryType()!=wkbMultiPolygon)//wkbGeometryType =6
    {
       OGR_F_Destroy( hFeat );
       num_seq++;
       continue;
    }
}
        }
        OGRGeometry *newShape;
        if(poShape->getGeometryType()==wkbPolygon)
            newShape=new OGRPolygon(*((OGRPolygon *) poShape));
        else if(poShape->getGeometryType()==wkbMultiPolygon)
            newShape=new OGRMultiPolygon(*((OGRMultiPolygon *) poShape));
        else
        {
            printf("unsuported geometry type, exiting.........\n");
            exit(-1);
        }

        polygon_vec.push_back(newShape);

        idx_vec.push_back(num_seq++);
        std::vector<double> aPointX;
        std::vector<double> aPointY;
        std::vector<int> aPartSize;
        PolygonFromGeometry( poShape, aPointX, aPointY, aPartSize );

        x_v.insert(x_v.end(),aPointX.begin(),aPointX.end());
        y_v.insert(y_v.end(),aPointY.begin(),aPointY.end());
        r_len_v.insert(r_len_v.end(),aPartSize.begin(),aPartSize.end());
        f_len_v.push_back(aPartSize.size());
        OGR_F_Destroy( hFeat );
        num_feature++;
    }
    g_len_v.push_back(num_feature);
    return num_feature;
}

void write_shapefile(const char * file_name,uint32_t num_poly,
    const double *x1,const double * y1,const double *x2,const double * y2)
{
    GDALAllRegister();
    const char *pszDriverName = "ESRI Shapefile";
    GDALDriver *poDriver= GetGDALDriverManager()->GetDriverByName(pszDriverName );

    if( poDriver == nullptr )
    {
        std::cout<<pszDriverName<<" driver not available."<<std::endl;
        exit( -1 );
    }

    GDALDataset * poDS = poDriver->Create( file_name, 0, 0, 0, GDT_Unknown, nullptr );
    if( poDS == nullptr )
    {
        std::cout<<"Creation of output file"<<file_name<<"  failed."<<std::endl;
        exit( -1 );
    }

    OGRLayer *poLayer= poDS->CreateLayer( "bbox", nullptr, wkbLineString, nullptr );
    if( poLayer == nullptr )
    {
        std::cout<<"Layer creation failed."<<std::endl;
        exit( -1 );
    }

    OGRFieldDefn oField0( "MID", OFTInteger );
    OGRFieldDefn oField1( "x1", OFTReal );
    OGRFieldDefn oField2( "y1", OFTReal );
    OGRFieldDefn oField3( "x2", OFTReal );
    OGRFieldDefn oField4( "y2", OFTReal );

    bool b0=(poLayer->CreateField( &oField0 ) != OGRERR_NONE);
    bool b1=(poLayer->CreateField( &oField1 ) != OGRERR_NONE);
    bool b2=(poLayer->CreateField( &oField2 ) != OGRERR_NONE);
    bool b3=(poLayer->CreateField( &oField3 ) != OGRERR_NONE);
    bool b4=(poLayer->CreateField( &oField4 ) != OGRERR_NONE);
    if(b0||b1||b2||b3||b4)
    {
       std::cout<<"Creating Name field failed."<<std::endl;
       exit( 1 );
    }
    uint32_t num_f=0;
    for(uint32_t i=0;i<num_poly;i++)
    {
        OGRFeature *poFeature=OGRFeature::CreateFeature( poLayer->GetLayerDefn() );
        assert(poFeature!=nullptr);
        poFeature->SetField( "MID",(int)i);
        poFeature->SetField( "x1",x1[i]);
        poFeature->SetField( "y1",y1[i]);
        poFeature->SetField( "x2",x2[i]);
        poFeature->SetField( "y2",y2[i]);

        OGRLineString *ls=(OGRLinearRing*)OGRGeometryFactory::createGeometry(wkbLinearRing);
        ls->addPoint(x1[i],y1[i]);
        ls->addPoint(x1[i],y2[i]);
        ls->addPoint(x2[i],y2[i]);
        ls->addPoint(x2[i],y1[i]);
        ls->addPoint(x1[i],y1[i]);

        OGRPolygon *polygon=(OGRPolygon*)OGRGeometryFactory::createGeometry(wkbPolygon);
        polygon->addRing(ls);
        poFeature->SetGeometry(polygon);

        if( poLayer->CreateFeature( poFeature ) != OGRERR_NONE )
        {
            std::cout<<"Failed to create feature in shapefile."<<std::endl;
            exit( -1 );
        }
        OGRFeature::DestroyFeature( poFeature );
        num_f++;
    }
    GDALClose( poDS );
}
//==========================================================================================================

/*
* helper c++ modules for testing spatial jion
*/

void polyvec_to_bbox(const std::vector<OGRGeometry *>& h_ogr_polygon_vec,const char * file_name,
    double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2)
{
    uint32_t num_poly=h_ogr_polygon_vec.size();

    h_x1=new double[num_poly];
    h_y1=new double[num_poly];
    h_x2=new double[num_poly];
    h_y2=new double[num_poly];
    assert(h_x1!=nullptr && h_y1!=nullptr && h_x2!=nullptr && h_y2!=nullptr);

    for(uint32_t i=0;i<num_poly;i++)
    {
        OGREnvelope env;
        h_ogr_polygon_vec[i]->getEnvelope(&env);
        h_x1[i]=env.MinX;
        h_y1[i]=env.MinY;
        h_x2[i]=env.MaxX;
        h_y2[i]=env.MaxY;
    }
    if(file_name==nullptr)
        return;
    FILE *fp=nullptr;
    if((fp=fopen(file_name,"w"))==nullptr)
    {
          std::cout<<"can not open "<<file_name<<"for output"<<std::endl;
          exit(-1);
    }
    for(uint32_t i=0;i<num_poly;i++)
        fprintf(fp,"%10d, %15.5f, %15.5f, %15.5f, %15.5f\n",i,h_x1[i],h_y1[i],h_x2[i],h_y2[i]);
    fclose(fp);
}



void rand_points_ogr_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const std::vector<OGRGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices)

{
    std::cout<<"h_ogr_polygon_vec.size()="<<h_ogr_polygon_vec.size()<<std::endl;
    h_pnt_idx_vec.clear();
    h_pnt_len_vec.clear();
    h_poly_idx_vec.clear();

    uint32_t num_samples=indices.size();
    char  msg[100];
    timeval t0,t1;
    gettimeofday(&t0, nullptr);

    for(uint32_t k=0;k<num_samples;k++)
    {
        uint32_t pntid=h_point_indices[indices[k]];
        OGRPoint pnt(h_pnt_x[pntid],h_pnt_y[pntid]);
        std::vector<uint32_t> temp_vec;
        for(uint32_t j=0;j<h_ogr_polygon_vec.size();j++)
        {
            if(h_ogr_polygon_vec[j]->Contains(&pnt))
                temp_vec.push_back(j);
        }
        if(temp_vec.size()>0)
        {
            h_pnt_len_vec.push_back(temp_vec.size());
            h_pnt_idx_vec.push_back(pntid);
            h_poly_idx_vec.insert(h_poly_idx_vec.end(),temp_vec.begin(),temp_vec.end());
        }
        if(k>0 && k%num_print_interval==0)
        {
            gettimeofday(&t1, nullptr);
            sprintf(msg,"loop=%d runtime for the last %d iterations is\n",k,num_print_interval);
            float cpu_time_per_interval=calc_time(msg,t0,t1);
            t0=t1;
         }
    }
    std::cout<<"h_pnt_len_vec.size="<<h_pnt_len_vec.size()<<std::endl;
}

void matched_pairs_ogr_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const uint32_t *h_pq_quad_idx,  const uint32_t *h_pq_poly_idx,
    const uint32_t *h_qt_length,  const uint32_t * h_qt_fpos,
    const std::vector<OGRGeometry *>& h_ogr_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices)
{
    h_pnt_idx_vec.clear();
    h_pnt_len_vec.clear();
    h_poly_idx_vec.clear();

    uint32_t num_samples=indices.size();
    char  msg[100];
    timeval t2,t3;
    uint32_t p=0;
    gettimeofday(&t2, nullptr);
    for(uint32_t k=0;k<num_samples;k++)
    {
        uint32_t qid=h_pq_quad_idx[indices[k]];
        uint32_t pid=h_pq_poly_idx[indices[k]];
        uint32_t qlen=h_qt_length[qid];
        uint32_t fpos=h_qt_fpos[qid];
        for(uint32_t i=0;i<qlen;i++)
        {
            assert(fpos+i<num_counts);
            OGRPoint pnt(h_pnt_x[fpos+i],h_pnt_y[fpos+i]);
            std::vector<uint32_t> temp_vec;
            if(h_ogr_polygon_vec[pid]->Contains(&pnt))
            {
                h_pnt_len_vec.push_back(1);
                uint32_t pntid=fpos+i;
                h_pnt_idx_vec.push_back(pntid);
                h_poly_idx_vec.push_back(pid);
            }
            if(p>0 && p%num_print_interval==0)
            {
                gettimeofday(&t3, nullptr);
                sprintf(msg,"loop=%d quad=%d runtime for the last %d iterations is\n",p,k,num_print_interval);
                float cpu_time_per_interval=calc_time(msg,t2,t3);
                t2=t3;
            }
           p++;
        }
    }
    std::cout<<"h_pnt_len_vec.size="<<h_pnt_len_vec.size()<<std::endl;
}

//copied from GEOS, required by initGEOS
static void notice(const char* fmt, ...)
{
    std::fprintf(stdout, "NOTICE: ");

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);

    std::fprintf(stdout, "\n");
}

void rand_points_geos_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const std::vector<GEOSGeometry *>& h_geos_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices)

{
    std::cout<<"h_geos_polygon_vec.size()="<<h_geos_polygon_vec.size()<<std::endl;
    h_pnt_idx_vec.clear();
    h_pnt_len_vec.clear();
    h_poly_idx_vec.clear();

    initGEOS(notice, notice);

    uint32_t num_samples=indices.size();
    char  msg[100];
    timeval t0,t1;
    gettimeofday(&t0, nullptr);

    for(uint32_t k=0;k<num_samples;k++)
    {
        uint32_t pntid=h_point_indices[indices[k]];
        GEOSCoordSequence* seq = GEOSCoordSeq_create(1, 2);
        GEOSCoordSeq_setX(seq, 0, h_pnt_x[pntid]);
        GEOSCoordSeq_setY(seq, 0, h_pnt_y[pntid]);
        GEOSGeometry *pnt=GEOSGeom_createPoint(seq);

        std::vector<uint32_t> temp_vec;
        for(uint32_t j=0;j<h_geos_polygon_vec.size();j++)
        {
             char res=GEOSContains(h_geos_polygon_vec[j],pnt);
            //std::cout<<"k="<<k<<" j="<<j<<" res="<<(uint32_t)res<<std::endl;
            if(res==1)
                temp_vec.push_back(j);
        }
        if(temp_vec.size()==1)
        {
   			h_pnt_len_vec.push_back(temp_vec.size());
            h_pnt_idx_vec.push_back(indices[k]);
            h_poly_idx_vec.insert(h_poly_idx_vec.end(),temp_vec.begin(),temp_vec.end());
        }
        else
           printf("multi: (%10.5f %10.5f) %d = pintid=%d size=%ld\n", h_pnt_x[k], h_pnt_y[k],k,pntid,temp_vec.size());
        if(k>0 && k%num_print_interval==0)
        {
            gettimeofday(&t1, nullptr);
            sprintf(msg,"loop=%d runtime for the last %d iterations is\n",k,num_print_interval);
            float cpu_time_per_interval=calc_time(msg,t0,t1);
            t0=t1;
         }
    }
    std::cout<<"h_pnt_len_vec.size="<<h_pnt_len_vec.size()<<std::endl;
    printf("rand_points_geos_pip_test:h_pnt_idx_vec................\n");
    std::copy(h_pnt_idx_vec.begin(),h_pnt_idx_vec.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    finishGEOS();
}

void matched_pairs_geos_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const uint32_t *h_pq_quad_idx,  const uint32_t *h_pq_poly_idx,
    const uint32_t *h_qt_length,  const uint32_t * h_qt_fpos,
    const std::vector<GEOSGeometry *>& h_geos_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y,const uint32_t *h_point_indices)
{
    h_pnt_idx_vec.clear();
    h_pnt_len_vec.clear();
    h_poly_idx_vec.clear();

    initGEOS(notice, notice);

    uint32_t num_samples=indices.size();
    char  msg[100];
    timeval t2,t3;
    uint32_t p=0;
    gettimeofday(&t2, nullptr);
    for(uint32_t k=0;k<num_samples;k++)
    {
        uint32_t qid=h_pq_quad_idx[indices[k]];
        uint32_t pid=h_pq_poly_idx[indices[k]];
        uint32_t qlen=h_qt_length[qid];
        uint32_t fpos=h_qt_fpos[qid];
        for(uint32_t i=0;i<qlen;i++)
        {
            assert(fpos+i<num_counts);
            GEOSCoordSequence* seq = GEOSCoordSeq_create(1, 2);
            GEOSCoordSeq_setX(seq, 0, h_pnt_x[fpos+i]);
            GEOSCoordSeq_setY(seq, 0, h_pnt_y[fpos+i]);
            GEOSGeometry *pnt=GEOSGeom_createPoint(seq);

            std::vector<uint32_t> temp_vec;
            //if(h_geos_polygon_vec[pid]->Contains(pnt))
            char res=GEOSContains(h_geos_polygon_vec[pid],pnt);
            if(res==1)
            {
                h_pnt_len_vec.push_back(1);
                uint32_t pntid=fpos+i;
                h_pnt_idx_vec.push_back(pntid);
                h_poly_idx_vec.push_back(pid);
            }
            if(p>0 && p%num_print_interval==0)
            {
                gettimeofday(&t3, nullptr);
                sprintf(msg,"loop=%d quad=%d runtime for the last %d iterations is\n",p,k,num_print_interval);
                float cpu_time_per_interval=calc_time(msg,t2,t3);
                t2=t3;
            }
           p++;
        }
    }
     std::cout<<"h_pnt_len_vec.size="<<h_pnt_len_vec.size()<<std::endl;
     finishGEOS();
}