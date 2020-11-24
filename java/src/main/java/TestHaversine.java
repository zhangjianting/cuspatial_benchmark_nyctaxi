import java.io.File;
import ai.rapids.cudf.*;
import ai.rapids.cuspatial.*;

public class TestHaversine
{
    public static void main(String[] args)
    {
      /*ColumnVector aLon = ColumnVector.fromDoubles(-180, 180);
      ColumnVector aLat = ColumnVector.fromDoubles(0, 30);
      ColumnVector bLon = ColumnVector.fromDoubles(180, -180);
      ColumnVector bLat = ColumnVector.fromDoubles(0, 30);*/

      /*['vendor_name', 'Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime',
       'Passenger_Count', 'Trip_Distance', 'Start_Lon', 'Start_Lat',
       'Rate_Code', 'store_and_forward', 'End_Lon', 'End_Lat', 'Payment_Type',
       'Fare_Amt',
      */

      //all parameters are place holders
      /*String csv_fn="";
      Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .column(DType.STRING, "D")
        .build();
      Table table = Table.readCSV(schema, new File(csv_fn));
      System.out.println(table.toString());*/

      String parquet_fn="/home/jianting/cuspatial_data/200901.parquet";
      Table table = Table.readCSV(new File(parquet_fn));

      //"Start_Lon"
      ColumnVector aLon=table.getColumn(0);
      //"Start_Lat"
      ColumnVector aLat=table.getColumn(1);
      //"End_Lon"
      ColumnVector bLon=table.getColumn(2);
      //"End_Lat"
      ColumnVector bLat=table.getColumn(3);

      ColumnVector result = CuSpatial.haversineDistance(aLon, aLat, bLon, bLat);
      HostColumnVector h_dis=result.copyToHost();
      System.out.println(h_dis.toString());
    }
}