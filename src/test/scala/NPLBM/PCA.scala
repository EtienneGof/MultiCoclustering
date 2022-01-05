package NPLBM

import Common.NormalInverseWishart
import Common.Tools.{getLoadingsAndVarianceExplained, getLoadingsAndEigenValues}
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.FunSuite

class PCA extends FunSuite {
  test("get loading") {
    {

      val multivariateGaussianModel = new NormalInverseWishart(10).sample()
      val covMat = multivariateGaussianModel.covariance
      val (loadings1, eigenValues) = getLoadingsAndVarianceExplained(covMat, 5, 0.99)

      assert(loadings1.rows == 5)
      assert(eigenValues.length == 5)

      val (_, eigenValues2) = getLoadingsAndVarianceExplained(covMat, 10, 0.6)
      val cumulatedVarExplained = eigenValues2.toArray.map{var s = 0D; d => {s += d; s}}

      assert(cumulatedVarExplained.last >= 0.6)

    }

  }
}