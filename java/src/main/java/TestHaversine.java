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

      //all parameters are place holders
      String csv_fn="";
      Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .column(DType.STRING, "D")
        .build();
      Table table = Table.readCSV(schema, new File(csv_fn));
      System.out.println(table.toString());

      //"Start_Lon"
      ColumnVector aLon=table.getColumn(5);
      //"Start_Lat"
      ColumnVector aLat=table.getColumn(6);
      //"End_Lon"
      ColumnVector bLon=table.getColumn(7);
      //"End_Lat"
      ColumnVector bLat=table.getColumn(8);

      ColumnVector result = CuSpatial.haversineDistance(aLon, aLat, bLon, bLat);
      HostColumnVector h_dis=result.copyToHost();
      System.out.println(h_dis.toString());
    }
}