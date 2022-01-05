package Common


import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.numerics.log
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.FunSuite

class ToolsTest extends FunSuite {
  test("log factorial") {
    {
      assertResult(logFactorial(5D))(log(factorial(5D)))
    }
  }
}