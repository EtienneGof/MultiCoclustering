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
    val interProductSumList: List[((Int,Int),  DenseMatrix[Double])] = internProductSumRDD.collect().toList

    interProductSumList.map(c => (c._1,c._2/(count(c._1)-1).toDouble)).toMap

  }

  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double], constraint: String = "none"): DenseMatrix[Double] = {

    require(List("none","independant").contains(constraint))
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

  def weightedCovariance (X: List[DenseVector[Double]],
                          weights: DenseVector[Double],
                          mode: DenseVector[Double],
                          constraint: String = "none"): DenseMatrix[Double] = {
    require(List("none","independant").contains(constraint))
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

  def meanListDV(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def weightedMean(X: List[DenseVector[Double]], weights: DenseVector[Double]): DenseVector[Double] = {
    require(X.length == weights.length)
    val res = X.indices.par.map(i => weights(i) * X(i)).reduce(_+_) / sum(weights)
    res
  }

  def sample(probabilities: List[Double]): Int = {
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

  def clustersParametersEstimation(Data: List[DenseVector[Double]],
                                   prior: NormalInverseWishart,
                                   rowPartition: List[Int]) : List[MultivariateGaussian] = {


    (Data zip rowPartition).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      val sufficientStatistic = prior.update(dataInCluster)
      (k, sufficientStatistic.expectation())
    }).toList.sortBy(_._1).map(_._2)

  }

  def clustersParametersEstimationLBM(dataByCol: List[List[DenseVector[Double]]],
                                      prior: NormalInverseWishart,
                                      rowPartition: List[Int],
                                      colPartition: List[Int]) : List[List[MultivariateGaussian]] = {

    (dataByCol zip colPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val l = e.head._2
      (l,
        (dataInCol.transpose zip rowPartition).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val k = f.head._2
          val sufficientStatistic = prior.update(dataInBlock)
          (k, sufficientStatistic.expectation())
        }).toList.sortBy(_._1).map(_._2))

    }).toList.sortBy(_._1).map(_._2)
  }

  def clustersParametersEstimationMC(dataByCol: List[List[DenseVector[Double]]],
                                     prior: NormalInverseWishart,
                                     rowPartitions: List[List[Int]],
                                     colPartition: List[Int])(implicit d: DummyImplicit) = {

    (dataByCol zip colPartition).groupBy(_._2).values.par.map(e => {
      val dataInCol: List[List[DenseVector[Double]]] = e.map(_._1)
      val l = e.head._2
      (l,
        (dataInCol.transpose zip rowPartitions(l)).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val k = f.head._2
          val sufficientStatistic = prior.update(dataInBlock)
          (k, sufficientStatistic.expectation())
        }).toList.sortBy(_._1).map(_._2))

    }).toList.sortBy(_._1).map(_._2)
  }

  def clustersParametersEstimationMCC(dataByCol: List[List[DenseVector[Double]]],
                                      prior: NormalInverseWishart,
                                      redundantColPartition: List[Int],
                                      correlatedColPartitions: List[List[Int]],
                                      rowPartitions: List[List[Int]]) = {

    (dataByCol zip redundantColPartition).groupBy(_._2).values.par.map(e => {
      val dataInRedundantCol: List[List[DenseVector[Double]]] = e.map(_._1)
      val h = e.head._2
      (h,
        (dataInRedundantCol zip correlatedColPartitions(h)).groupBy(_._2).values.par.map(f => {
          val dataInCorrelatedCol: List[List[DenseVector[Double]]] = f.map(_._1)
          val l = f.head._2
          (l,
            (dataInCorrelatedCol.transpose zip rowPartitions(h)).groupBy(_._2).values.par.map(e => {
              val dataInBlock = e.map(_._1).reduce(_++_)
              val k = e.head._2
              val sufficientStatistic = prior.update(dataInBlock)
              (k, sufficientStatistic.expectation())
            }).toList.sortBy(_._1).map(_._2))
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def clustersParametersEstimationDPV(dataByCol: List[List[DenseVector[Double]]],
                                      priors: List[NormalInverseWishart],
                                      rowPartitions: List[List[Int]],
                                      colPartition: List[Int])(implicit d: DummyImplicit) = {

    dataByCol.indices.par.map(j => {
      (j,
        (dataByCol(j) zip rowPartitions(colPartition(j))).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1)
          val k = f.head._2
          val sufficientStatistic = priors(j).update(dataInBlock)
          (k, sufficientStatistic.expectation())
        }).toList.sortBy(_._1).map(_._2))

    }).toList.sortBy(_._1).map(_._2)
  }


  def sample(nCategory: Int, weight: List[Double] = Nil): Int = {
    val finalWeight = if(weight==Nil){
      List.fill(nCategory)(1D/nCategory.toDouble)
    } else {
      weight
    }
    sample(finalWeight)
  }

  def sampleWithSeed(probabilities: List[Double], seed: Long ): Int = {
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

  def scale(data: List[List[DenseVector[Double]]]): List[List[DenseVector[Double]]] = {
    data.map(column => {
      val mode = meanListDV(column)
      val cov = covariance(column,mode,"independant")
      val std = sqrt(diag(cov))
      column.map(_ /:/ std)
    })
  }

  def weight(data: List[List[DenseVector[Double]]], weight: DenseVector[Double]): List[List[DenseVector[Double]]] = {
    data.map(column => {
      column.map(_ *:* weight)
    })
  }

  def intToVecProb(i: Int, size:Int): List[Double] = {
    val b = mutable.Buffer.fill(size)(1e-8)
    b(i)=1D
    val sum = b.sum
    b.map(_/sum) .toList
  }

  def partitionToBelongingProbabilities(partition: List[Int], toLog:Boolean=false): List[List[Double]]={

    val K = partition.max+1
    val res = partition.indices.map(i => {
      intToVecProb(partition(i),K)
    }).toList

    if(!toLog){res} else {res.map(_.map(log(_)))}
  }

  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def logSumExp(X: DenseVector[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def normalizeProbability(probs: List[Double]): List[Double] = {
    normalizeLogProbability(probs.map(e => log(e)))
  }

  def normalizeLogProbability(probs: List[Double]): List[Double] = {
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

  def mapDm(probBelonging: DenseMatrix[Double]): List[Int] = {
    probBelonging(*,::).map(argmax(_)).toArray.toList
  }

  def MAP(probBelonging: List[List[Double]]): List[Int] = {
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

    max(if(sample(List(pi, 1 - pi)) == 0){
      Gamma(shape = shape + nCluster, newScale).draw()
    } else {
      Gamma(shape = shape + nCluster - 1, newScale).draw()
    }, 1e-8)
  }

  def stickBreaking(hyperPrior: Gamma, size:Int, seed : Option[Int] = None): List[Double] = {

    val actualSeed: Int = seed match {
      case Some(s) => s
      case _ => sample(1000000)
    }

    implicit val Rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(actualSeed)))
    val concentrationParam = hyperPrior.draw()

    val betaPrior = new Beta(1,concentrationParam)(Rand)
    val betaDraw: List[Double] = (0 until size).map(_ => betaPrior.draw()).toList

    val pi: List[Double] = if(size == 1){
      betaDraw
    } else {
      betaDraw.head +: (1 until betaDraw.length).map(j => {
        betaDraw(j) * betaDraw.dropRight(betaDraw.length - j).map(1 - _).product
      }).toList
    }
    pi
  }

}
