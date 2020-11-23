//javac -cp /home/jianting/.m2/repository/ai/rapids/cuspatial/0.16-SNAPSHOT/cuspatial-0.16-SNAPSHOT-cuda10-1.jar:/home/jianting/.m2/repository/ai/rapids/cudf/0.16-SNAPSHOT/cudf-0.16-SNAPSHOT-cuda10-1.jar TestHaversine.java

// /usr/lib/jvm/java-11-openjdk-amd64/bin/java -cp /home/jianting/.m2/repository/ai/rapids/cuspatial/0.16-SNAPSHOT/cuspatial-0.16-SNAPSHOT-cuda10-1.jar:/home/jianting/.m2/repository/ai/rapids/cudf/0.16-SNAPSHOT/cudf-0.16-SNAPSHOT-cuda10-1.jar:/home/jianting/cuspatial_benchmark_nyctaxi/java TestHaversine

// /usr/lib/jvm/java-11-openjdk-amd64/bin/java -cp /tmp:/home/jianting/cuspatial_benchmark_nyctaxi/java  TestHaversine

package ai.rapids.cuspatial_benchmark;

import ai.rapids.cudf.*;
import ai.rapids.cuspatial.*;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

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