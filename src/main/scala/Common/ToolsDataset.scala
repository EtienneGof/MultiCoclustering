package Common

import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax => bzArgMax}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object ToolsDataset extends java.io.Serializable {


  def dataWithProbFromDataWithPartitions(data: Dataset[Row], K:Int, L:Int)(implicit ss: SparkSession): Dataset[(Int, Int, Int, DenseVector[Double], DenseVector[Double])] = {

    import ss.implicits._

    data.map(r =>
      (r.getInt(0), r.getInt(1), r.getInt(2),
        DenseVector(intToVecProb(r.getInt(3), K)),
        DenseVector(intToVecProb(r.getInt(4), L)))
    )
  }

  def joinLBMBiPartitionToDataCells(data: Dataset[Row],
                                    rowPartition: List[Int],
                                    colPartition: List[Int])(implicit ss: SparkSession): Dataset[Row] = {

    import ss.sqlContext.implicits._

    ss.sql("SELECT name, age FROM data2 WHERE age BETWEEN 13 AND 19")

  }
}
