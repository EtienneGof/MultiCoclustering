package DPMM

import Common.NormalInverseWishart
import Common.ProbabilisticTools._
import Common.Tools._
import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

class Default(Data: Array[DenseVector[Double]],
              prior: NormalInverseWishart = new NormalInverseWishart(),
              alphaPrior: Gamma,
              initByUserPartition: Option[Array[Int]] = None) extends Serializable {

  val n: Int = Data.length

  val priorPredictive: Array[Double] = Data.map(prior.predictive)

  var partition: Array[Int] = initByUserPartition match {
    case Some(m) =>
      require(m.length == Data.length)
      m
    case None => Array.fill(Data.length)(0)
  }

  var countCluster: ArrayBuffer[Int] = partitionToOrderedCount(partition).to[ArrayBuffer]


  var (actualAlpha, updateAlphaFlag): (Double, Boolean) = (alphaPrior.mean, true)


  def conditionalAlphaUpdate(): Unit = {
    if(updateAlphaFlag){actualAlpha = updateAlpha(actualAlpha, alphaPrior, countCluster.length, n)}
  }

  def posteriorPredictive(data: DenseVector[Double], cluster: Int): Double = {0D}

  def computeClusterPartitionProbabilities(idx: Int,
                                            verbose: Boolean=false): Array[Double] = {
    countCluster.indices.map(k => {
      (k, log(countCluster(k).toDouble) + posteriorPredictive(Data(idx), k))
    }).toArray.sortBy(_._1).map(_._2)
  }

  def drawMembership(i: Int): Unit = {
    val probPartition = computeClusterPartitionProbabilities(i)
    val probPartitionNewCluster = log(actualAlpha) + priorPredictive(i)
    val normalizedProbs = normalizeLogProbability(probPartition :+ probPartitionNewCluster)
    val newPartition = sample(normalizedProbs)
    partition = partition.updated(i, newPartition)
  }

  def removeElementFromCluster(idx: Int): Unit = {
    val currentPartition = partition(idx)
    if (countCluster(currentPartition) == 1) {
      countCluster.remove(currentPartition)
      partition = partition.map(c => { if( c > currentPartition ){ c - 1 } else c })
    } else {
      countCluster.update(currentPartition, countCluster.apply(currentPartition) - 1)
    }
  }

  def addElementToCluster(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      countCluster = countCluster ++ ArrayBuffer(1)
    } else {
      countCluster.update(newPartition, countCluster.apply(newPartition) + 1)
    }
  }

  def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToCluster(i)
    }
  }

}
