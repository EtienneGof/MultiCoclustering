package Common

import Common.Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, diag, max, sum}
import breeze.numerics.{exp, log, sqrt}
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object ProbabilisticTools extends java.io.Serializable {

  def variance(X: DenseVector[Double]): Double = {
    covariance(X,X)
  }

  def covariance(X: DenseVector[Double],Y: DenseVector[Double]): Double = {
    sum( (X- meanDV(X)) *:* (Y- meanDV(Y)) ) / (Y.length-1)
  }

  def covarianceSpark(X: RDD[((Int, Int), Vector)],
                      modes: Map[(Int, Int), DenseVector[Double]],
                      count: Map[(Int, Int), Int]): Map[(Int, Int),  DenseMatrix[Double]] = {

    val XCentered : RDD[((Int, Int), DenseVector[Double])] = X.map(d => (d._1, DenseVector(d._2.toArray) - DenseVector(modes(d._1).toArray)))

    val internProduct = XCentered.map(row => (row._1, row._2 * row._2.t))
    val internProductSumRDD: RDD[((Int,Int), DenseMatrix[Double])] = internProduct.reduceByKey(_+_)
    val interProductSumArray: Array[((Int,Int),  DenseMatrix[Double])] = internProductSumRDD.collect()

    interProductSumArray.map(c => (c._1,c._2/(count(c._1)-1).toDouble)).toMap

  }

  def covariance(X: Array[DenseVector[Double]], mode: DenseVector[Double], constraint: String = "none"): DenseMatrix[Double] = {

    require(Array("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    val p = XMat.cols
    val covmat = constraint match {
      case "independant" => DenseMatrix.tabulate[Double](p,p){(i, j) => if(i == j) covariance(XMat(::,i),XMat(::,i)) else 0D}
      case _ => {
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        XMatCentered.t * XMatCentered
      }/ (X.length.toDouble-1)
    }

    roundMat(covmat,8)
  }

  def weightedCovariance(X: DenseVector[Double], Y: DenseVector[Double], weights: DenseVector[Double]): Double = {
    sum( weights *:* (X- meanDV(X)) *:* (Y- meanDV(Y)) ) / sum(weights)
  }

  def weightedCovariance (X: Array[DenseVector[Double]],
                          weights: DenseVector[Double],
                          mode: DenseVector[Double],
                          constraint: String = "none"): DenseMatrix[Double] = {
    require(Array("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    require(weights.length==X.length)

    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    //    val p = XMat.cols
    val q = mode.length
    constraint match {
      case "independant" => DenseMatrix.tabulate[Double](q,q){(i, j) => if(i == j) weightedCovariance(XMat(::,i),XMat(::,i), weights) else 0D}
      case _ =>
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        val res = DenseMatrix((0 until XMatCentered.rows).par.map(i => {
          weights(i)*XMatCentered(i,::).t * XMatCentered(i,::)
        }).reduce(_+_).toArray:_*)/ sum(weights)
        res.reshape(q,q)
    }
  }

  def meanDV(X: DenseVector[Double]): Double = {
    sum(X)/X.length
  }

  def meanArrayDV(X: Array[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def weightedMean(X: Array[DenseVector[Double]], weights: DenseVector[Double]): DenseVector[Double] = {
    require(X.length == weights.length)
    val res = X.indices.par.map(i => weights(i) * X(i)).reduce(_+_) / sum(weights)
    res
  }

  def sample(probabilities: Array[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def clustersParametersEstimation(Data: Array[DenseVector[Double]],
                                   prior: NormalInverseWishart,
                                   rowPartition: Array[Int]) : Array[MultivariateGaussian] = {


    (Data zip rowPartition).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      val sufficientStatistic = prior.update(dataInCluster)
      (k, sufficientStatistic.sample())
    }).toArray.sortBy(_._1).map(_._2)

  }

  def clustersParametersEstimation(dataByCol: Array[Array[DenseVector[Double]]],
                                   prior: NormalInverseWishart,
                                   rowPartition: Array[Int],
                                   colPartition: Array[Int]) : Array[Array[MultivariateGaussian]] = {

    (dataByCol zip colPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val l = e.head._2
      (l,
        (dataInCol.transpose zip rowPartition).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val k = f.head._2
          val sufficientStatistic = prior.update(dataInBlock)
          (k, sufficientStatistic.expectation())
        }).toArray.sortBy(_._1).map(_._2))

    }).toArray.sortBy(_._1).map(_._2)
  }

  def clustersParametersEstimationMC(dataByCol: Array[Array[DenseVector[Double]]],
                                     prior: NormalInverseWishart,
                                     rowPartitions: Array[Array[Int]],
                                     colPartition: Array[Int])(implicit d: DummyImplicit): Array[Array[MultivariateGaussian]] = {

    (dataByCol zip colPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val l = e.head._2
      (l,
        (dataInCol.transpose zip rowPartitions(l)).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val k = f.head._2
          val sufficientStatistic = prior.update(dataInBlock)
          (k, sufficientStatistic.sample())
        }).toArray.sortBy(_._1).map(_._2))

    }).toArray.sortBy(_._1).map(_._2)
  }

  def clustersParametersEstimationMCC(dataByCol: Array[Array[DenseVector[Double]]],
                                      prior: NormalInverseWishart,
                                      redundantColPartition: Array[Int],
                                      correlatedColPartitions: Array[Array[Int]],
                                      rowPartitions: Array[Array[Int]]): Array[Array[MultivariateGaussian]] = {

    val combinedColPartition = Common.Tools.combineRedundantAndCorrelatedColPartitions(redundantColPartition, correlatedColPartitions)

    val rowPartitionDuplicatedPerColCluster = correlatedColPartitions.indices.map(h => {
      Array.fill(correlatedColPartitions(h).distinct.length)(rowPartitions(h))
    }).reduce(_++_)

    clustersParametersEstimationMC(dataByCol, prior, rowPartitionDuplicatedPerColCluster, combinedColPartition)
  }

  def clustersParametersEstimationDPV(dataByCol: Array[Array[DenseVector[Double]]],
                                      priors: Array[NormalInverseWishart],
                                      rowPartitions: Array[Array[Int]],
                                      colPartition: Array[Int])(implicit d: DummyImplicit): Array[Array[MultivariateGaussian]] = {

    dataByCol.indices.par.map(j => {
      (j,
        (dataByCol(j) zip rowPartitions(colPartition(j))).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1)
          val k = f.head._2
          val sufficientStatistic = priors(j).update(dataInBlock)
          (k, sufficientStatistic.sample())
        }).toArray.sortBy(_._1).map(_._2))

    }).toArray.sortBy(_._1).map(_._2)
  }


  def sample(nCategory: Int, weight: Array[Double] = Array.emptyDoubleArray): Int = {
    val finalWeight = if(weight.isEmpty){
      Array.fill(nCategory)(1D/nCategory.toDouble)
    } else {
      weight
    }
    sample(finalWeight)
  }

  def sampleWithSeed(probabilities: Array[Double], seed: Long ): Int = {
    val dist = probabilities.indices zip probabilities
    val r = new scala.util.Random(seed)
    val threshold = r.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def scale(data: Array[Array[DenseVector[Double]]]): Array[Array[DenseVector[Double]]] = {
    data.map(column => {
      val mode = meanArrayDV(column)
      val cov = covariance(column,mode,"independant")
      val std = sqrt(diag(cov))
      column.map(_ /:/ std)
    })
  }

  def weight(data: Array[Array[DenseVector[Double]]], weight: DenseVector[Double]): Array[Array[DenseVector[Double]]] = {
    data.map(column => {
      column.map(_ *:* weight)
    })
  }

  def intToVecProb(i: Int, size:Int): Array[Double] = {
    val b = mutable.Buffer.fill(size)(1e-8)
    b(i)=1D
    val sum = b.sum
    b.map(_/sum) .toArray
  }

  def partitionToBelongingProbabilities(partition: Array[Int], toLog:Boolean=false): Array[Array[Double]]={

    val K = partition.max+1
    val res = partition.indices.map(i => {
      intToVecProb(partition(i),K)
    }).toArray

    if(!toLog){res} else {res.map(_.map(log(_)))}
  }

  def logSumExp(X: Array[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def logSumExp(X: DenseVector[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def normalizeProbability(probs: Array[Double]): Array[Double] = {
    normalizeLogProbability(probs.map(e => log(e)))
  }

  def normalizeLogProbability(probs: Array[Double]): Array[Double] = {
    val LSE = Common.ProbabilisticTools.logSumExp(probs)
    probs.map(e => exp(e - LSE))
  }

  def unitCovFunc(fullCovarianceMatrix:Boolean):
  DenseVector[Double] => DenseMatrix[Double] = (x: DenseVector[Double]) => {
    if(fullCovarianceMatrix){
      x * x.t
    } else {
      diag(x *:* x)
    }
  }

  def mapDm(probBelonging: DenseMatrix[Double]): Array[Int] = probBelonging(*, ::).map(argmax(_)).toArray

  def MAP(probBelonging: Array[Array[Double]]): Array[Int] = {
    probBelonging.map(Tools.argmax)
  }

  def updateAlpha(alpha: Double, alphaPrior: Gamma, nCluster: Int, nObservations: Int): Double = {
    val shape = alphaPrior.shape
    val rate =  1D / alphaPrior.scale

    val log_x = log(new Beta(alpha + 1, nObservations).draw())
    val pi1 = shape + nCluster + 1
    val pi2 = nObservations * (rate - log_x)
    val pi = pi1 / (pi1 + pi2)
    val newScale = 1 / (rate - log_x)

    max(if(sample(Array(pi, 1 - pi)) == 0){
      Gamma(shape = shape + nCluster, newScale).draw()
    } else {
      Gamma(shape = shape + nCluster - 1, newScale).draw()
    }, 1e-8)
  }

  def stickBreaking(hyperPrior: Gamma, size:Int, seed : Option[Int] = None): Array[Double] = {

    val actualSeed: Int = seed match {
      case Some(s) => s
      case _ => sample(1000000)
    }

    implicit val Rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(actualSeed)))
    val concentrationParam = hyperPrior.draw()

    val betaPrior = new Beta(1,concentrationParam)(Rand)
    val betaDraw: Array[Double] = (0 until size).map(_ => betaPrior.draw()).toArray

    val pi: Array[Double] = if(size == 1){
      betaDraw
    } else {
      betaDraw.head +: (1 until betaDraw.length).map(j => {
        betaDraw(j) * betaDraw.dropRight(betaDraw.length - j).map(1 - _).product
      }).toArray
    }
    pi
  }

}
