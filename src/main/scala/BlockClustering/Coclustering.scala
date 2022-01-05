package BlockClustering

import Common.NormalInverseWishart
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec

class Coclustering(val DataByRow: List[List[DenseVector[Double]]],
                   var alphaRow: Option[Double] = None,
                   var alphaRowPrior: Option[Gamma] = None,
                   var alphaCol: Option[Double] = None,
                   var alphaColPrior: Option[Gamma] = None,
                   var initByUserPrior: Option[NormalInverseWishart] = None,
                   var initByUserRowPartition: Option[List[Int]] = None,
                   var initByUserColPartition: Option[List[Int]] = None) {

  val prior: NormalInverseWishart = initByUserPrior match {
    case Some(pr) => pr
    case None => new NormalInverseWishart(DataByRow)
  }

  var rowClustering = new Clustering(DataByRow, alphaRow, alphaRowPrior, Some(prior), initByUserRowPartition, initByUserColPartition)
  var colClustering = new Clustering(DataByRow.transpose, alphaCol, alphaColPrior, Some(prior),
    Some(rowClustering.colPartition), Some(rowClustering.rowPartition), Some(rowClustering.NIWParamsByRow.transpose))


  def computeLikelihood: Double = {
    val components = rowClustering.parametersEstimation
    val rowLikelihood = rowClustering.likelihood(components)

    rowLikelihood + colClustering.probabilityPartition
  }

  def run(nIter: Int = 10,
          verbose: Boolean = false,
          printLikelihood: Boolean = false): (List[List[Int]], List[List[Int]], List[List[List[MultivariateGaussian]]], List[Double]) = {

    var likelihoodEveryIteration = List(computeLikelihood)

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Coclustering Iteration: " + iter.toString)
        Common.Tools.prettyPrintLBM(rowClustering.countRowCluster.toList, rowClustering.countColCluster.toList)
        Common.Tools.prettyPrintLBM(colClustering.countRowCluster.toList, colClustering.countColCluster.toList)
      }

      if (iter <= nIter) {

        var t0 = System.nanoTime()

        rowClustering.setColPartitionAndNIWParams(colClustering.rowPartition, colClustering.NIWParamsByRow.transpose)
        rowClustering.updatePartition()

        if(verbose){
          t0 = printTime(t0, "draw row Partition")
        }

        colClustering.setColPartitionAndNIWParams(rowClustering.rowPartition, rowClustering.NIWParamsByRow.transpose)
        colClustering.updatePartition()

        if(verbose){
          t0 = printTime(t0, "draw col Partition")
        }

        likelihoodEveryIteration =  likelihoodEveryIteration ++
          List(computeLikelihood)

        go(iter + 1)
      }
    }

    go(1)

    (rowClustering.rowPartitionEveryIteration,
      colClustering.rowPartitionEveryIteration,
      rowClustering.componentsEveryIterations,
      likelihoodEveryIteration)

  }

}
