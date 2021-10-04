package Common

import Common.Tools.allEqual
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.numerics.sin
import breeze.stats.distributions.{Gaussian, MultivariateGaussian}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.math._
import scala.util.Random

object functionPrototypes  {

  // Sin prototype
  def f1(x: List[Double]): List[Double] = {
    x.map(t => sin(6 * scala.math.Pi * t) / 2)
  }

  // Sigmoid prototype
  def f2(x: List[Double]): List[Double] = {
    val center = 0.6
    val slope = 40D
    val maxVal =  1D
    x.map(t => maxVal / (1 + exp(-slope * (t - center))))
  }

  // Rectangular prototype
  def f3(x: List[Double]): List[Double] = {
    val start = 0.3
    val duration = 0.3
    x.map({
      case t if t < `start` || t >= (`start`+`duration`) => 0D
      case _ => 1D
    })
  }

  // Morlet prototype
  def f4(x: List[Double]): List[Double] = {
    val center = 0.5
    x.map(t => {
      val u = (t - center) * 10
      exp(-0.5 * pow(u,2)) * cos(5 * u)
    })
  }

  def gaussianProto(x: List[Double], center: Double = 0.6, sd:Double =0.03): List[Double] = {
    val G = Gaussian(center, sd)
    x.map(t => G.pdf(t) / G.pdf(center))
  }

  // Gaussian prototype
  def f5(x: List[Double]): List[Double] = {
    gaussianProto(x)
  }

  // Double Gaussian prototype
  def f6(x: List[Double]): List[Double] = {

    val c1 = gaussianProto(x, 0.3, 0.2)
    val c2 = gaussianProto(x, 0.7)

    (0.7 * DenseVector(c1.toArray) + 0.3 * DenseVector(c2.toArray)).toArray.toList
  }

  // y-shifted Gaussian prototype
  def f7(x: List[Double]): List[Double] = {
    x.map(t => 0.3 + 0.3 * sin(2 * scala.math.Pi * t))
  }

  // sin by rectangular prototype
  def f8(x: List[Double]): List[Double] = {
    val start = 0.4
    val duration = 0.3
    x.map({
      case t if t <`start` || t>(`start`+`duration`) => -0.7 + 0.1* sin(40 * Pi * t)
      case t => 0.7 + 0.05 * sin(50 * Pi *t)
    })
  }

  // sin by square function
  def f9(x: List[Double]): List[Double] = {
    x.map(t => 0.02 * sin(30 * (t + 0.08)) / ((t + 0.08) * (t + 0.08)))
  }

}
