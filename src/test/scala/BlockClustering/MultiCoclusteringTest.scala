package BlockClustering

import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gamma
import org.scalatest.FunSuite

import scala.collection.mutable.ListBuffer
import scala.util.Random

class MultiCoclusteringTest extends FunSuite {

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
    sizeClusterCol = List(15, 15))).transpose

  val Data2: List[List[DenseVector[Double]]] = matrixToDataByCol(Common.DataGeneration.randomLBMDataGeneration(
    modesByCol = modes2,
    covariancesByCol = covariances2,
    sizeClusterRow = List(20, 40),
    sizeClusterCol = List(20, 10))).transpose

  val Data: List[List[DenseVector[Double]]] = Data1 ++ Data2
  val n = 60
  val p = 45

}
