package Common

import breeze.linalg.{DenseMatrix, DenseVector, det, inv, sum, trace}
import breeze.numerics.constants.Pi
import breeze.numerics.{log, multiloggamma, pow}
import breeze.stats.distributions.{MultivariateGaussian, Wishart, Gamma => GammaDistrib}
import org.apache.commons.math3.special.Gamma

class NormalInverseWishart(var mu: DenseVector[Double] = DenseVector(0D),
                           var kappa: Double = 1D,
                           var psi: DenseMatrix[Double] = DenseMatrix(1D),
                           var nu: Int = 1
                          ) {
  var d: Int = psi.rows
  var studentNu: Int = this.nu - d + 1
  var studentPsi: DenseMatrix[Double] = ((this.kappa + 1) / (this.kappa * studentNu)) * this.psi

  def this(dataArray: Array[Array[DenseVector[Double]]])= {
    this()
    val dataFlattened = dataArray.reduce(_++_)
    val globalMean     = Common.ProbabilisticTools.meanArrayDV(dataFlattened)
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
    val newSig = Wishart(this.nu, this.psi).sample()
    val newMu = MultivariateGaussian(this.mu, inv(newSig * this.kappa)).draw()
    MultivariateGaussian(newMu, newSig/pow(this.kappa,2))
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

  def jointPosteriorPredictive(newObs: Array[DenseVector[Double]], X: Array[DenseVector[Double]]): Double = {
    this.update(X).jointPriorPredictive(newObs)
  }

  def jointPriorPredictive(X: Array[DenseVector[Double]]): Double = {
    val m = X.length
    val updatedPrior = this.update(X)
    val a = -m * d * 0.5 * log(Pi)
    val b = (d / 2D) * log(this.kappa / updatedPrior.kappa)
    val c = multiloggamma(updatedPrior.nu / 2D, d) - multiloggamma(this.nu / 2D, d)
    val e =  (this.nu / 2D) * log(det(this.psi)) - (updatedPrior.nu / 2D) * log(det(updatedPrior.psi))
    a + b + c + e
  }

  def update(data: Array[DenseVector[Double]]): NormalInverseWishart = {

    val n = data.length.toDouble
    val meanData = data.reduce(_ + _) / n.toDouble
    val newKappa: Double = this.kappa + n
    val newNu = this.nu + n.toInt
    val newMu = (this.kappa.toDouble * this.mu + n * meanData) / newKappa
    val x_mu0 = meanData - this.mu

    val C = if (n == 1) {
      DenseMatrix.zeros[Double](d,d)
    } else {
      sum(data.map(x => {
        val x_mu = x - meanData
        x_mu * x_mu.t
      }).toList)
    }

    val newPsi = this.psi + C + ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def weightedUpdate(data: DenseVector[Double], weight: Int): NormalInverseWishart = {
    val repeatedData = Array.fill(weight)(data)
    val n = repeatedData.length.toDouble
    val meanData = repeatedData.reduce(_ + _) / n.toDouble
    val newKappa: Double = kappa + n.toDouble
    val newNu = this.nu + n.toInt
    val newMu = (this.kappa * this.mu + n * meanData) / newKappa
    val x_mu0 = meanData - this.mu
    val newPsi = this.psi + ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }

  def checkNIWParameterEquals (SS2: NormalInverseWishart): Boolean = {
    (mu == SS2.mu) & (nu == SS2.nu) & ((psi - SS2.psi).toArray.sum <= 10e-7) & (kappa == SS2.kappa)
  }

  def print(): Unit = {
    println()
    println("mu: " + mu.toString)
    println("kappa: " + kappa.toString)
    println("psi: " + psi.toString)
    println("nu: " + nu.toString)
    println()
  }

  def removeObservations(data: Array[DenseVector[Double]]): NormalInverseWishart = {
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
      }).toList)
    }
    val newPsi = this.psi - C - ((n * this.kappa) / newKappa) * (x_mu0 * x_mu0.t)
    new NormalInverseWishart(newMu, newKappa, newPsi, newNu)
  }


  def weightedRemoveObservation(data: DenseVector[Double], weight: Int): NormalInverseWishart = {
    val repeatedData = Array.fill(weight)(data)

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
      }).toList)
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
                           countByCluster: Array[Int],
                           n: Int): Double = {
    nCluster * log(alpha) +
      countByCluster.map(c => Common.Tools.logFactorial(c - 1)).sum -
      (0 until n).map(e => log(alpha + e.toDouble)).sum
  }


  def DPMMLikelihood(alpha: Double,
                     data: Array[DenseVector[Double]],
                     membership: Array[Int],
                     countCluster: Array[Int],
                     components: Array[MultivariateGaussian]): Double = {
    val K = countCluster.length
    val partitionDensity = probabilityPartition(K, alpha, countCluster, data.length)
    val dataLikelihood = data.indices.map( i => components(membership(i)).logPdf(data(i))).sum
    val paramsDensity = components.map(logPdf).sum
    partitionDensity + paramsDensity + dataLikelihood
  }

  def NPLBMlikelihood(alphaRowPrior: GammaDistrib,
                      alphaColPrior: GammaDistrib,
                      alphaRow: Double,
                      alphaCol: Double,
                      dataByCol: Array[Array[DenseVector[Double]]],
                      rowMembership: Array[Int],
                      colMembership: Array[Int],
                      countRowCluster: Array[Int],
                      countColCluster: Array[Int],
                      componentsByCol: Array[Array[MultivariateGaussian]]): Double = {

    val K = countRowCluster.length
    val L = countColCluster.length
    val alphaRowDensity = alphaRowPrior.logPdf(alphaRow)
    val alphaColDensity = alphaColPrior.logPdf(alphaCol)
    val rowPartitionDensity = probabilityPartition(K, alphaRow, countRowCluster, dataByCol.head.length)
    val colPartitionDensity = probabilityPartition(L, alphaCol, countColCluster, dataByCol.length)
    val dataLikelihood = dataByCol.indices.par.map(j => {
      dataByCol.head.indices.map(i => {
        componentsByCol(colMembership(j))(rowMembership(i)).logPdf(dataByCol(j)(i))
      }).sum
    }).sum
    val paramsDensity = componentsByCol.reduce(_++_).map(logPdf).sum
    alphaRowDensity + alphaColDensity + rowPartitionDensity + colPartitionDensity + paramsDensity + dataLikelihood
  }

  def NPLBMlikelihood(alphaRowPrior: GammaDistrib,
                      alphaColPrior: GammaDistrib,
                      alphaRow: Double,
                      alphaCol: Double,
                      dataByCol: Array[Array[DenseVector[Double]]],
                      rowMembership: Array[Int],
                      colMembership: Array[Int],
                      countRowCluster: Array[Int],
                      countColCluster: Array[Int]): Double = {

    val K = countRowCluster.length
    val L = countColCluster.length
    val alphaRowDensity = alphaRowPrior.logPdf(alphaRow)
    val alphaColDensity = alphaColPrior.logPdf(alphaCol)
    val rowPartitionDensity = probabilityPartition(K, alphaRow, countRowCluster, dataByCol.head.length)
    val colPartitionDensity = probabilityPartition(L, alphaCol, countColCluster, dataByCol.length)

    val dataLikelihood =
      (dataByCol zip colMembership).groupBy(_._2).values.par.map(e => {
        val dataInCol = e.map(_._1).transpose
        (dataInCol zip rowMembership).groupBy(_._2).values.par.map(elts => {
          val dataInBlock = elts.map(_._1)
          jointPriorPredictive(dataInBlock.flatten)
        }).sum
      }).sum

    alphaRowDensity + alphaColDensity + rowPartitionDensity + colPartitionDensity + dataLikelihood
  }


  def MCCLikelihood(alphaPrior: GammaDistrib,
                    betaPrior: GammaDistrib,
                    gammaPrior: GammaDistrib,
                    alphas: List[Double],
                    betas: List[Double],
                    gamma: Double,
                    dataByCol: Array[Array[DenseVector[Double]]],
                    redundantColMembership: Array[Int],
                    correlatedColMembership: Array[Array[Int]],
                    rowMemberships: Array[Array[Int]],
                    countRedundantColCluster: Array[Int],
                    countCorrelatedColCluster: Array[Array[Int]],
                    countRowCluster: Array[Array[Int]]): Double = {

    val H = countRowCluster.length

    val gammaDensity = gammaPrior.logPdf(gamma)

    val NPLBMLikelihoods = (0 until H).map(h => {
        NPLBMlikelihood(alphaPrior, betaPrior, alphas(h), betas(h), dataByCol,
          rowMemberships(h), correlatedColMembership(h),countRowCluster(h), countCorrelatedColCluster(h))
    }).sum

    val redundantColPartitionDensity = probabilityPartition(H, gamma, countRedundantColCluster, countRedundantColCluster.length)

    gammaDensity + redundantColPartitionDensity + NPLBMLikelihoods
  }

  def posteriorSample(Data: Array[DenseVector[Double]],
                      rowMembership: Array[Int]) : Array[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).sample())
    }).toArray.sortBy(_._1).map(_._2)
  }

  def posteriorSample(DataByCol: Array[Array[DenseVector[Double]]],
                      rowMembership: Array[Int],
                      colMembership: Array[Int]) : Array[Array[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_++_)
          val l = f.head._2
          (l, this.update(dataInBlock).sample())
        }).toArray.sortBy(_._1).map(_._2))
    }).toArray.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(Data: Array[DenseVector[Double]],
                           rowMembership: Array[Int]) : Array[MultivariateGaussian] = {
    (Data zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCluster = e.map(_._1)
      val k = e.head._2
      (k, this.update(dataInCluster).expectation())
    }).toArray.sortBy(_._1).map(_._2)
  }

  def posteriorExpectation(DataByCol: Array[Array[DenseVector[Double]]],
                           rowMembership: Array[Int],
                           colMembership: Array[Int]) : Array[Array[MultivariateGaussian]] = {
    (DataByCol.transpose zip rowMembership).groupBy(_._2).values.par.map(e => {
      val dataInCol = e.map(_._1)
      val k = e.head._2
      (k,
        (dataInCol.transpose zip rowMembership).groupBy(_._2).values.par.map(f => {
          val dataInBlock = f.map(_._1).reduce(_ ++ _)
          val l = f.head._2
          (l, this.update(dataInBlock).expectation())
        }).toArray.sortBy(_._1).map(_._2))
    }).toArray.sortBy(_._1).map(_._2)
  }

  def expectation() : MultivariateGaussian = {
    val expectedSigma = this.psi / (this.nu - d - 1).toDouble
    val expectedMu = this.mu
    MultivariateGaussian(expectedMu, expectedSigma)
  }

}
