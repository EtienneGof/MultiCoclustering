import BlockClustering.{Clustering, Coclustering}
import Common.Tools.{matrixToDataByCol, partitionToOrderedCount}
import breeze.linalg.{DenseMatrix, DenseVector}

object coclusteringExample {

  def main(args: Array[String]) {
    val modes = List(List(DenseVector(1.9, 1.9), DenseVector(1.4, -1.4)),
      List(DenseVector(-1.7, 1.7),DenseVector(-1.7, -1.7)),
      List(DenseVector(-3, -3D),DenseVector(5, 5D)))

    val covariances = List(
      List(DenseMatrix(0.25, 0.2, 0.2, 0.25).reshape(2,2),
        DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2)),
      List(DenseMatrix(0.12, 0.0, 0.0, 0.12).reshape(2,2),
        DenseMatrix(0.3, 0.0, 0.0, 0.3).reshape(2,2)),
      List(DenseMatrix(0.22, 0.0, 0.0, 0.15).reshape(2,2),
        DenseMatrix(0.1, 0.0, 0.0, 0.1).reshape(2,2)))

    val Data: List[List[DenseVector[Double]]] = matrixToDataByCol(Common.DataGeneration.randomLBMDataGeneration(
      modesByCol = modes,
      covariancesByCol = covariances,
      sizeClusterRow = List(30, 30),
      sizeClusterCol = List(15, 15, 15))).transpose

    val coclustering = new Coclustering(Data, alphaRow = Some(10D), alphaCol = Some(10D))

    val (rowPartitionEveryIteration, colPartitionEveryIteration, componentsEveryIterations, likelihoodEveryIteration) = coclustering.run(20)

    println("The resulting row x column cluster count should equal the data generation inputs: (30, 30) x (15, 15, 15)")

    println("Estimated number of elements by columns, rows, and blocks: \n")
    Common.Tools.prettyPrintLBM(partitionToOrderedCount(rowPartitionEveryIteration.last),
      partitionToOrderedCount(colPartitionEveryIteration.last))

  }

}
