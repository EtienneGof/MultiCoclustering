import BlockClustering.Clustering
import Common.Tools.partitionToOrderedCount
import breeze.linalg.{DenseMatrix, DenseVector}

object clusteringExample {

  def main(args: Array[String]) {

    val modes = List(DenseVector(1.9, 1.9), DenseVector(1.4, -1.4), DenseVector(-1.7, 1.7), DenseVector(-1.7, -1.7))

    val covariances = List(
      DenseMatrix(0.25, 0.2, 0.2, 0.25).reshape(2,2),
      DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2),
      DenseMatrix(0.12, 0.0, 0.0, 0.12).reshape(2,2),
      DenseMatrix(0.3, 0.0, 0.0, 0.3).reshape(2,2))

    val data = Common.DataGeneration.randomMixture(modes, covariances, List(50, 50, 50, 50)).map(List(_))
    val clustering = new Clustering(data, alpha = Some(5D))
    val (rowPartitionEveryIteration,  _, _) = clustering.run(100)

    println(rowPartitionEveryIteration.last)
    println("The resulting cluster count should be around List(50, 50, 50, 50)")
    println(partitionToOrderedCount(rowPartitionEveryIteration.last))
  }

}
