package Common

import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.FunSuite

class NormalInverseWishartTest extends FunSuite {
  test("Update NIW parameters with 1 observation") {
    {

      val data1 = List(DenseVector(1D))
      val data2 = List(DenseVector(2D), DenseVector(3D))
      val data3 = List(DenseVector(1D), DenseVector(2D), DenseVector(3D))

      val initDistribution = new NormalInverseWishart()
      val distribution1 = initDistribution.update(data1)
      val distribution2 = initDistribution.update(data2)
      val distribution3 = initDistribution.update(data3)

      assert(distribution1.update(data2).checkNIWParameterEquals(distribution3))
      assert(distribution3.removeObservations(data1).checkNIWParameterEquals(distribution2))

    }
  }

  test("Update NIW parameters with several elements") {
    {
      val data1 = List(DenseVector(1D), DenseVector(2D), DenseVector(3D), DenseVector(4D))
      val data2 = List(DenseVector(1D), DenseVector(2D))
      val data3 = List(DenseVector(3D), DenseVector(4D))
      val initParams = new NormalInverseWishart()
      val params1 = initParams.update(data1)
      val params2 = initParams.update(data2)
      assert(params2.update(data3).checkNIWParameterEquals(params1))
      assert(params1.removeObservations(data3).checkNIWParameterEquals(params2))
    }
  }

  test("Update NIW with data") {
    {

      val n = 100
      val data = MultivariateGaussian(DenseVector(0D, 0D), DenseMatrix.eye[Double](2)).sample(n).toList
      val globalMean = Common.Tools.mean(data)
      val globalVariance = Common.Tools.covariance(data, globalMean)
      val globalPrecision = inv(globalVariance)
      val prior = new NormalInverseWishart(globalMean, 1D, globalPrecision, data.head.length + 1)
      assertResult(globalMean)(prior.mu)
      assertResult(globalPrecision)(prior.psi)

      val updatedPrior = prior.update(data)

      assertResult(1D + n)(updatedPrior.kappa)
      assertResult(data.head.length + 1 + n)(updatedPrior.nu)

    }
  }

  test("Normal Inverse Wishart density") {
    {
      val prior = new NormalInverseWishart(
        DenseVector(0.2, 0.3),
        2,
        DenseMatrix(Array(3.1, 0D, 0D, 1.35)).reshape(2, 2),
        5)
      val mv = MultivariateGaussian(DenseVector(1, 0),
        1.2 * DenseMatrix.eye[Double](2))
      val d = prior.logPdf(mv)
      assertResult(-5.9921409)(round(d, 7))
    }
  }
}
