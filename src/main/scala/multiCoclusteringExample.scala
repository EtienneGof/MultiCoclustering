import BlockClustering.{MultiCoclustering}
import Common.Tools.{matrixToDataByCol, partitionToOrderedCount}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gamma

object multiCoclusteringExample {

  def main(args: Array[String]) {
    val modes1 = List(List(DenseVector(1.9, 1.9), DenseVector(1.4, -1.4)),
      List(DenseVector(-1.7, 1.7),DenseVector(-1.7, -1.7)))

    val modes2 = List(List(DenseVector(1.9, 1.9), DenseVector(1.4, -1.4)),
      List(DenseVector(-3, -3D),DenseVector(5, 5D)))

    val covariances1 = List(
      List(DenseMatrix(0.25, 0.2, 0.2, 0.25).reshape(2,2),
        DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2)),
      List(DenseMatrix(0.12, 0.0, 0.0, 0.12).reshape(2,2),
        DenseMatrix(0.3, 0.0, 0.0, 0.3).reshape(2,2)))

    val covariances2 = List(
      List(DenseMatrix(0.25, 0.2, 0.2, 0.25).reshape(2,2),
        DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2)),
      List(DenseMatrix(0.22, 0.0, 0.0, 0.15).reshape(2,2),
        DenseMatrix(0.1, 0.0, 0.0, 0.1).reshape(2,2)))

    val Data1: List[List[DenseVector[Double]]] = matrixToDataByCol(Common.DataGeneration.randomLBMDataGeneration(
      modesByCol = modes1,
      covariancesByCol = covariances1,
      sizeClusterRow = List(30, 30),
      sizeClusterCol = List(15, 15)))

    val Data2: List[List[DenseVector[Double]]] = matrixToDataByCol(Common.DataGeneration.randomLBMDataGeneration(
      modesByCol = modes2,
      covariancesByCol = covariances2,
      sizeClusterRow = List(20, 40),
      sizeClusterCol = List(20, 10)))

    val Data: List[List[DenseVector[Double]]] = (Data1 ++ Data2).transpose


    val alphaPrior = Gamma(shape = 10, scale = 5)
    val betaPrior = Gamma(shape = 100, scale = 200)
    val gammaPrior = Gamma(shape = 10, scale = 1)

    val MCC = new MultiCoclustering(Data, alphaPrior, betaPrior, gammaPrior)

    val (varPartition, rowPartitions, columnPartitions) = MCC.run(20, verbose = true)

    println()
    println("The resulting block count should equal the data generation inputs: \n" +
      "       - 1 coclustering structure with dimension (30, 30) x (15, 15)\n" +
      "       - 1 coclustering structure with dimension (20, 40) x (10, 20)")

    println("Number of elements by columns, rows, and blocks: \n")

    Common.Tools.prettyPrintMCC(rowPartitions.map(partitionToOrderedCount),
      columnPartitions.map(partitionToOrderedCount))

  }

}
