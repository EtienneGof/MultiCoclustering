package Common

import breeze.linalg.{DenseMatrix, DenseVector, det, inv, sum, trace}
import breeze.numerics.constants.Pi
import breeze.numerics.{log, multiloggamma, pow}
import breeze.stats.distributions.{MultivariateGaussian, Wishart}
import org.apache.commons.math3.special.Gamma

class NormalInverseWishart(var mu: DenseVector[Double] = DenseVector(0D),
                           var kappa: Double = 1D,
                           var psi: DenseMatrix[Double] = DenseMatrix(1D),
                           var nu: Int = 1
                          ) {
  var d: Int = psi.rows
  var studentNu: Int = this.nu - d + 1
  var studentPsi: DenseMatrix[Double] = ((this.kappa + 1) / (this.kappa * studentNu)) * this.psi

  def this(dimension: Int) = {
    this(DenseVector(Array.fill(dimension)(0D)),
      1D,
      DenseMatrix.eye(dimension),
      dimension + 1)
    this
  }

  def this(dataList: List[DenseVector[Double]])(implicit x: DummyImplicit) = {
    this()
    val globalMean     = Common.ProbabilisticTools.meanListDV(dataList)
    val globalVariance = Common.ProbabilisticTools.covariance(dataList, globalMean)
    val globalPrecision = inv(globalVariance)
    this.mu = globalMean
    this.kappa = 1D
    this.psi = globalPrecision
    this.nu = globalMean.length + 1
    this.d = psi.rows
    this.studentNu = this.nu - d + 1
    this.studentPsi = ((this.kappa + 1) / (this.kappa * studentNu)) * this.psi
  }


  def this(dataList: List[List[DenseVector[Double]]]) = {
    this()
    val dataFlattened = dataList.reduce(_++_)
    val globalMean     = Common.ProbabilisticTools.meanListDV(dataFlattened)
    val globalVariance = Common.ProbabilisticTools.covariance(dataFlattened, globalMean)
    val globalPrecision = inv(globalVariance)
    this.mu = globalMean
    this.kappa = 1D
    this.psi = globalPrecision
    this.nu = globalMean.length + 1
    this.d = psi.rows
    this.studentNu = this.nu - d + 1
    this.studentPsi = ((this.kappa + 1) / (this.kappa * studentNu)) * this.psi
  }

  def sample(): MultivariateGaussian = {
    try {
      val newSig = Wishart(this.nu, this.psi).sample()
      val newMu = MultivariateGaussian(this.mu, Common.Tools.roundMat(inv(newSig * this.kappa), 8)).draw()
      MultivariateGaussian(newMu, newSig/pow(this.kappa,2))
    } catch {
      case e: Exception  => {
        println(e)
        val newSig = Wishart(this.nu, this.psi).sample()
        val newMu = MultivariateGaussian(this.mu, Common.Tools.roundMat(inv(newSig * this.kappa), 8)).draw()
        MultivariateGaussian(newMu, newSig/pow(this.kappa,2))
      }
    }
  }

  def multivariateStudentLogPdf(x: DenseVector[Double], mu: DenseVector[Double], sigma: DenseMatrix[Double], nu: Double): Double = {
    val d = mu.length
    val x_mu = x - mu
    val a = Gamma.logGamma((nu + d) / 2D)
    val b = Gamma.logGamma(nu / 2D) + (d / 2D) * log(Pi * nu) + .5 *log(det(sigma))
    val c = -(nu + d) / 2D * log(1 + (1D / nu)* (x_mu.t * inv(sigma) * x_mu))
    a - b + c
  }

  def predictive(x: DenseVector[Double]): Double = {
    this.multivariateStudentLogPdf(x, this.mu, studentPsi, studentNu)
  }

  def jointPosteriorPredictive(newObs: List[DenseVector[Double]], X: List[DenseVector[Double]]): Double = {
    this.update(X).jointPriorPredictive(newObs)
  }

  def jointPriorPredictive(X: List[DenseVector[Double]]): Double = {
    val m = X.length
    val updatedPrior = this.update(X)
    val a = -m * d * 0.5 * log(Pi)
    val b = (d / 2D) * log(this.kappa / updatedPrior.kappa)
    val c = multiloggamma(updatedPrior.nu / 2D, d) - multiloggamma(this.nu / 2D, d)
    val e =  (this.nu / 2D) * log(det(this.psi)) - (updatedPrior.nu / 2D) * log(det(updatedPrior.psi))
    a + b + c + e
  }

  def update(data: List[DenseVector[Double]]): NormalInverseWishart = {

    val n = data.length.toDouble
    val meanData = data.reduce(_ + _) / n.toDouble
    val newKappa: Double = this.kappa + n
    val newNu = this.nu + n.toInt
    val newMu = (this.kappa * this.mu + n * meanData) / newKappa
    val x_mu0 = meanData - this.mu

    val C = if (n == 1) {
      DenseMatrix.zeros[Double](d,d)
    } else {
      sum(data.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }))
    }

    val newPsi = this.psi + C + ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def weightedUpdate(data: DenseVector[Double], weight: Int): NormalInverseWishart = {
    val repeatedData = List.fill(weight)(data)
    val n = repeatedData.length.toDouble
    val meanData = repeatedData.reduce(_ + _) / n.toDouble
    val newKappa: Double = this.kappa + n
    val newNu = this.nu + n.toInt
    val newMu = (this.kappa * this.mu + n * meanData) / newKappa
    val x_mu0 = meanData - this.mu
    val newPsi = this.psi + ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def checkNIWParameterEquals (SS2: NormalInverseWishart): Boolean = {
    ((mu - SS2.mu).toArray.sum <= 10e-7) & (nu == SS2.nu) & ((psi - SS2.psi).toArray.sum <= 10e-7) & (kappa == SS2.kappa)
  }

  def print(): Unit = {
    println()
    println("mu: " + mu.toString)
    println("kappa: " + kappa.toString)
    println("psi: " + psi.toString)
    println("nu: " + nu.toString)
    println()
  }

  def removeObservations(data: List[DenseVector[Double]]): NormalInverseWishart = {
    val n = data.length.toDouble
    val meanData = data.reduce(_ + _) / n.toDouble
    val newKappa = this.kappa - n
    val newNu = this.nu - n.toInt
    val newMu = (this.kappa * this.mu - n * meanData) / newKappa
    val x_mu0 = meanData - this.mu
    val C = if (n == 1) {
      DenseMatrix.zeros[Double](d,d)
    } else {
      sum(data.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }))
    }
    val newPsi = this.psi - C - ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }


  def weightedRemoveObservation(data: DenseVector[Double], weight: Int): NormalInverseWishart = {
    val repeatedData = List.fill(weight)(data)

    val n = weight.toDouble
    val meanData = repeatedData.reduce(_ + _) / n.toDouble
    val newKappa = this.kappa - n
    val newNu = this.nu - n.toInt
    val newMu = (this.kappa * this.mu - n * meanData) / newKappa
    val x_mu0 = meanData - this.mu
    val C = if (n == 1) {
      DenseMatrix.zeros[Double](d,d)
    } else {
      sum(repeatedData.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }))
    }

    val newPsi = this.psi - C - ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    require(newKappa>0)
    require(newNu>0)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def logPdf(multivariateGaussian: MultivariateGaussian): Double = {
    val gaussianLogDensity = MultivariateGaussian(this.mu, multivariateGaussian.covariance/this.kappa).logPdf(multivariateGaussian.mean)
    val invWishartLogDensity = InvWishartlogPdf(multivariateGaussian.covariance)
    gaussianLogDensity + invWishartLogDensity
  }

  def InvWishartlogPdf(Sigma:DenseMatrix[Double]): Double = {
    (nu / 2D) * log(det(psi)) -
      ((nu * d) / 2D) * log(2) -
      multiloggamma( nu / 2D, d) -
      0.5*(nu + d + 1) * log(det(Sigma)) -
      0.5 * trace(psi * inv(Sigma))
  }

  def probabilityPartition(nCluster: Int,
                           alpha: Double,
                           countByCluster: List[Int],
                           n: Int): Double = {
    nCluster * log(alpha) +
      countByCluster.map(c => Common.Tools.logFactorial(c - 1)).sum -
      (0 until n).map(e => log(alpha + e.toDouble)).sum
  }


  def DPMMLikelihood(alpha: Double,
                     data: List[DenseVector[Double]],
                     membership: List[Int],
                     countCluster: List[Int],
                     components: List[MultivariateGaussian]): Double = {
    val K = countCluster.length
    val partitionDensity = probabilityPartition(K, alpha, countCluster, data.length)
    val dataLikelihood = data.indices.map( i => components(membership(i)).logPdf(data(i))).sum
    val paramsDensity = components.map(logPdf).sum
    partitionDensity + paramsDensity + dataLikelihood
  }

  def NPLBMlikelihood(alphaRow: Double,
                      alphaCol: Double,
                      dataByCol: List[List[DenseVector[Double]]],
                      rowMembership: List[Int],
                      colMembership: List[Int],
                      countRowCluster: List[Int],
                      countColCluster: List[Int],
                      componentsByCol: List[List[MultivariateGaussian]]): Double = {

    val K = countRowCluster.length
    val L = countColCluster.length

    val rowPartitionDensity = probabilityPartition(K, alphaRow, countRowCluster, dataByCol.head.length)
    val colPartitionDensity = probabilityPartition(L, alphaCol, countColCluster, dataByCol.length)

    val dataLikelihood = dataByCol.indices.par.map(j => {
      dataByCol.head.indices.map(i => {
        componentsByCol(colMembership(j))(rowMembership(i)).logPdf(dataByCol(j)(i))
      }).sum
    }).sum

    val paramsDensity = componentsByCol.reduce(_++_).map(logPdf).sum
    rowPartitionDensity + colPartitionDensity + paramsDensity + dataLikelihood
  }

  def NPCLBMDPVlikelihood(alphaRowPrior: breeze.stats.distributions.Gamma,
                          alphaColPrior: breeze.stats.distributions.Gamma,
                          alphaRows: List[Double],
                          alphaCol: Double,
                          dataByCol: List[List[DenseVector[Double]]],
                          rowMembership: List[List[Int]],
                          colMembership: List[Int],
                          countRowCluster: List[List[Int]],
                          countColCluster: List[Int],
                          priors: List[NormalInverseWishart],
                          componentsByVar: List[List[MultivariateGaussian]]): Double = {

    val Ks = countRowCluster.map(_.length)
    val L = countColCluster.length

    require(rowMembership.length == countRowCluster.length)
    require(componentsByVar.length == dataByCol.length)

    val alphaDensity = alphaRows.map(alphaRowPrior.logPdf).sum + alphaColPrior.logPdf(alphaCol)

    val colPartitionDensity = probabilityPartition(L, alphaCol, countColCluster, dataByCol.length)
    val rowPartitionDensity = alphaRows.indices.map(l => probabilityPartition(Ks(l),alphaRows(l), countRowCluster(l), dataByCol.head.length)).sum

    val dataLikelihood = dataByCol.indices.par.map(j => {
      dataByCol.head.indices.map(i => {
        val comp = componentsByVar(j)(rowMembership(colMembership(j))(i))
        comp.logPdf(dataByCol(j)(i))
      }).sum
    }).sum

    val paramsDensity = componentsByVar.indices.map(j => {
      componentsByVar(j).map(priors(j).logPdf).sum
    }).sum

    rowPartitionDensity + colPartitionDensity + paramsDensity + dataLikelihood + alphaDensity
  }


  def posteriorSample(Data: List[DenseVector[Double]],
                      rowMembership: List[Int]) : List[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).sample())
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorSample(DataByCol: List[List[DenseVector[Double]]],
                      rowMembership: List[Int],
                      colMembership: List[Int]) : List[List[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val l = f.head._2
          (l, this.update(dataInBlock).sample())
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(Data: List[DenseVector[Double]],
                           rowMembership: List[Int]) : List[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).expectation())
    }).toList.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(DataByCol: List[List[DenseVector[Double]]],
                           rowMembership: List[Int],
                           colMembership: List[Int]) : List[List[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_ ++ _)
          val l = f.head._2
          (l, this.update(dataInBlock).expectation())
        }).toList.sortBy(_._1).map(_._2))
    }).toList.sortBy(_._1).map(_._2)
  }

  def expectation() : MultivariateGaussian = {
    val expectedSigma = this.psi / (this.nu - d - 1).toDouble
    val expectedMu = this.mu
    MultivariateGaussian(expectedMu, expectedSigma)
  }

}
