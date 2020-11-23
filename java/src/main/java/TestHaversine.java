import ai.rapids.cudf.*;
import ai.rapids.cuspatial.*;

public class TestHaversine
{
    public static void main(String[] args)
    {
      ColumnVector aLon = ColumnVector.fromDoubles(-180, 180);
      ColumnVector aLat = ColumnVector.fromDoubles(0, 30);
      ColumnVector bLon = ColumnVector.fromDoubles(180, -180);
      ColumnVector bLat = ColumnVector.fromDoubles(0, 30);
      ColumnVector result = CuSpatial.haversineDistance(aLon, aLat, bLon, bLat);
      HostColumnVector h_dis=result.copyToHost();
      System.out.println(h_dis.toString());
    }
}