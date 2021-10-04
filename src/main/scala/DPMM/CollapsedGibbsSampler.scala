package DPMM

import Common.NormalInverseWishart
import Common.Tools._
import Common.ProbabilisticTools._
import breeze.linalg.{DenseVector, inv}
import breeze.numerics.log
import breeze.stats.distributions.{Beta, Gamma, MultivariateGaussian}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

class CollapsedGibbsSampler(Data: Array[DenseVector[Double]],
                            prior: NormalInverseWishart = new NormalInverseWishart(),
                            alphaPrior: Gamma,
                            initByUserPartition: Option[Array[Int]] = None
                           ) extends Default(Data, prior, alphaPrior, initByUserPartition) {

  var NIWParams: ArrayBuffer[NormalInverseWishart] = (Data zip partition).groupBy(_._2).values.map(e => {
    val dataPerCluster = e.map(_._1)
    val clusterIdx = e.head._2
    (clusterIdx, prior.update(dataPerCluster))
  }).toArray.sortBy(_._1).map(_._2).to[ArrayBuffer]

  override def posteriorPredictive(observation: DenseVector[Double], cluster: Int): Double = {
    NIWParams(cluster).predictive(observation)
  }

  def removeElementFromNIW(idx: Int): Unit = {
    val currentPartition =  partition(idx)
    if (countCluster(currentPartition) == 1) {
      NIWParams.remove(currentPartition)
    } else {
      val updatedNIWParams = NIWParams(currentPartition).removeObservations(Array(Data(idx)))
      NIWParams.update(currentPartition, updatedNIWParams)
    }
  }

  def addElementToNIW(idx: Int): Unit = {
    val newPartition = partition(idx)
    if (newPartition == countCluster.length) { // Creation of a new cluster
      val newNIWparam = this.prior.update(Array(Data(idx)))
      NIWParams = NIWParams ++ ArrayBuffer(newNIWparam)
    } else {
      val updatedNIWParams = NIWParams(newPartition).update(Array(Data(idx)))
      NIWParams.update(newPartition, updatedNIWParams)
    }
  }

  override def updatePartition(verbose: Boolean = false): Unit = {
    for (i <- 0 until n) {
      removeElementFromNIW(i)
      removeElementFromCluster(i)
      drawMembership(i)
      addElementToNIW(i)
      addElementToCluster(i)
    }
  }
  
  def run(maxIter: Int = 10,
          verbose: Boolean = false): (Array[Array[Int]], Array[Array[MultivariateGaussian]], Array[Double]) = {

    var membershipEveryIteration = Array(partition)
    var componentEveryIteration = Array(prior.posteriorSample(Data, partition).map(e => MultivariateGaussian(e.mean, inv(e.covariance))))
    var likelihoodEveryIteration = Array(prior.DPMMLikelihood(actualAlpha, Data, partition, countCluster.toArray, componentEveryIteration.head))

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n>>>>>> Iteration: " + iter.toString)
        println("\u03B1 = " + actualAlpha.toString)
        println("Cluster sizes: "+ countCluster.mkString(" "))
      }

      if (iter <= maxIter) {

        updatePartition()

        conditionalAlphaUpdate()

        val components = prior.posteriorSample(Data, partition)
        val likelihood = prior.DPMMLikelihood(actualAlpha,
          Data,
          partition,
          countCluster.toArray,
          components)
        membershipEveryIteration = membershipEveryIteration :+ partition
        componentEveryIteration = componentEveryIteration :+ components
        likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
        go(iter + 1)
      }
    }

    go(1)

    (membershipEveryIteration, componentEveryIteration, likelihoodEveryIteration)
  }
}

