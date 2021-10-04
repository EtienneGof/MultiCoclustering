package Common

import Common.ProbabilisticTools.unitCovFunc
import breeze.linalg.eigSym.EigSym
import breeze.linalg.{*, DenseMatrix, DenseVector, eigSym, max, min, sum}
import breeze.numerics.{abs, exp, log, sqrt}
import breeze.stats.distributions.RandBasis
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Matrices, Matrix}
import org.apache.spark.rdd.RDD
import smile.validation.{NormalizedMutualInformation, adjustedRandIndex, randIndex}

import scala.collection.immutable
import scala.util.{Success, Try}

object Tools extends java.io.Serializable {

  def filterMatrixByColumn(Mat: DenseMatrix[DenseVector[Double]], indexList: List[Int]): DenseMatrix[DenseVector[Double]] = {
    val dataList = indexList.map(Mat(*, ::).map(_.toArray.toList).toArray.toList.transpose).transpose
    DenseMatrix(dataList: _*)
  }

  //  def relabel(L: List[Int]): List[Int] = {
  //    val uniqueLabels = L.distinct.sorted
  //    val dict = uniqueLabels.zipWithIndex.toMap
  //    L.map(dict)
  //  }

  def relabel[T: Ordering](L: Array[T]): Array[Int] = {
    val uniqueLabels = L.distinct.sorted
    val dict = uniqueLabels.zipWithIndex.toMap
    L.map(dict)
  }

  def checkPosDef(M: DenseMatrix[Double]): Unit = {
    val EigSym(lambda, _) = eigSym(M)
    assert(lambda.forall(_>0))
  }

  def partitionToOrderedCount(partition: Array[Int]): Array[Int] = {
    partition.groupBy(identity).mapValues(_.length).toArray.sortBy(_._1).map(_._2)
  }


  def prettyFormatLBM(countRowCluster: Array[Int], countColCluster: Array[Int]): DenseMatrix[String] = {
    val mat: DenseMatrix[String] = (DenseVector(countRowCluster) * DenseVector(countColCluster).t).map(i => i.toString)

    val rowName: DenseMatrix[String] = DenseMatrix.vertcat(
      DenseMatrix(countRowCluster.map(_.toString)),
      DenseMatrix(Array.fill(countRowCluster.length)("|"))).t

    val colName: DenseMatrix[String] = DenseMatrix.horzcat(
      DenseMatrix.vertcat(
        DenseMatrix(Array(" "," ")),
        DenseMatrix(Array(" "," "))
      ),
      DenseMatrix.vertcat(
        DenseMatrix(countColCluster.map(_.toString)),
        DenseMatrix(Array.fill(countColCluster.length)("—"))
      )
    )

    DenseMatrix.vertcat(
      colName,
      DenseMatrix.horzcat(rowName, mat)
    )
  }

  def prettyPrintLBM(countRowCluster: Array[Int], countColCluster: Array[Int]): Unit = {
    println(prettyFormatLBM(countRowCluster, countColCluster))
  }

  def prettyPrintCLBM(countRowCluster: Array[Array[Int]], countColCluster: Array[Int]): Unit = {

    val L = countRowCluster.length
    val K_max = countRowCluster.map(_.length).max
    val mat = DenseMatrix.tabulate[String](L,K_max){
      case (i, j) => if(countRowCluster(i).length > j){
        (countRowCluster(i)(j)*countColCluster(i)).toString} else {"-"}

    }
    println(mat.t)

  }

  def prettyPrint(sizePerBlock: Map[(Int,Int), Int]): Unit = {
    val keys = sizePerBlock.keys
    val L = keys.map(_._1).max + 1
    val K = keys.map(_._2).max + 1
    val mat = DenseMatrix.tabulate[String](L,K){
      case (i, j) => if(sizePerBlock.contains((i,j))){
        sizePerBlock(i,j).toString } else {"-"}

    }
    println(mat.t)
  }

  def nestedMap[A, B, C](listA: List[List[A]],listB: List[List[B]])(f: (A,B) => C): List[List[C]] = {
    require(listA.length == listB.length)
    listA.indices.foreach(i => require(listA(i).length== listB(i).length))
    listA.indices.map(i => listA(i).indices.map(j => f(listA(i)(j), listB(i)(j))).toList).toList
  }

  def getPartitionFromSize(size: List[Int]): List[Int] = {
    size.indices.map(idx => List.fill(size(idx))(idx)).reduce(_ ++ _)
  }

  def aggregateListListDV(a: List[List[DenseVector[Double]]],
                          b: List[List[DenseVector[Double]]]): List[List[DenseVector[Double]]] = {
    a.indices.map(l => {
      a(l).indices.map(k => {
        a(l)(k) + b(l)(k)
      }).toList
    }).toList
  }

  def sortedCount(l: List[Int]): List[Int] = {
    l.groupBy(identity).mapValues(_.size).toList.sortBy(_._1).map(_._2)
  }

  def aggregateListDV(a: List[DenseVector[Double]],
                      b: List[DenseVector[Double]]): List[DenseVector[Double]] = {
    a.indices.map(i => {
      a(i) + b(i)
    }).toList
  }

  def getProportions(l: List[Int]): List[Double] = {
    val countCols = l.groupBy(identity).mapValues(_.size)
    countCols.map(c => c._2 / countCols.values.sum.toDouble).toList
  }

  def argmax(l: Array[Double]): Int ={
    l.view.zipWithIndex.maxBy(_._1)._2
  }

  def getTime[R](block: => R): Double = {
    val t0 = System.nanoTime()
    // call-by-name
    val t1 = System.nanoTime()
    (t1 - t0)/ 1e9
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/ 1e9 + "s")
    result
  }

  def printTime(t0:Long, stepName: String, verbose:Boolean = true): Long = {
    if(verbose){
      println(stepName.concat(" step duration: ").concat(((System.nanoTime - t0)/1e9D ).toString))
    }
    System.nanoTime
  }

  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) : Traversable[(X,Y)] = for { x <- xs; y <- ys } yield (x, y)
  }

  def denseMatrixToMatrix(A: DenseMatrix[Double]): Matrix = {
    Matrices.dense(A.rows, A.cols, A.toArray)
  }

  def checkPartitionEqual(partitionA : List[Int], partitionB: List[Int]): Boolean = {
    require(partitionA.length == partitionB.length)
    val uniqueA = partitionA.distinct.zipWithIndex
    val uniqueB = partitionB.distinct.zipWithIndex
    val dictA = (for ((k, v) <- uniqueA) yield (v, k)).toMap
    val dictB = (for ((k, v) <- uniqueB) yield (v, k)).toMap
    val reOrderedA = partitionA.map(e => dictA(e))
    val reOrderedB = partitionB.map(e => dictB(e))

    sum(DenseVector(reOrderedA.toArray)-DenseVector(reOrderedB.toArray))==0
  }

  def matrixToDenseMatrix(A: Matrix): DenseMatrix[Double] = {
    val p = A.numCols
    val n = A.numRows
    DenseMatrix(A.toArray).reshape(n,p)
  }

  def isInteger(x: String) : Boolean = {
    val y = Try(x.toInt)
    y match {
      case Success(_) => true
      case _ => false
    }
  }

  def inverseIndexedList(data: List[(Int, Array[DenseVector[Double]])]): List[(Int, Array[DenseVector[Double]])] = {
    val p = data.take(1).head._2.length
    (0 until p).map(j => {
      (j, data.map(row => row._2(j)).toArray)
    }).toList
  }

  def getEntireRowPartition(rowPartition: List[List[Int]]): Array[Int] = {
    val n = rowPartition.head.length
    val L = rowPartition.length
    val rowMultiPartition: List[List[Int]] = (0 until n).map(i => (0 until L).map(l => rowPartition(l)(i)).toList).toList

    val mapMultiPartitionToRowCluster = rowMultiPartition.distinct.zipWithIndex.toMap
    rowMultiPartition.map(mapMultiPartitionToRowCluster(_)).toArray
  }

  def remove[T](list: List[T], idx: Int):List[T] = list.patch(idx, Nil, 1)

  def insert[T](list: List[T], i: Int, values: T*): List[T] = {
    val (front, back) = list.splitAt(i)
    front ++ values ++ back
  }

  def insertList[T](array: Array[T], idx: Int, values: Array[T]): List[T] = {
    val (front, back) = array.splitAt(idx)
    front.toList ++ values.toList ++ back.toList
  }

  def roundMat(m: DenseMatrix[Double], digits:Int=0): DenseMatrix[Double] = {
    m.map(round(_,digits))
  }

  def roundDv(m: DenseVector[Double], digits:Int=0): DenseVector[Double] = {
    m.map(round(_,digits))
  }

  def allEqual[T](x: List[T], y:List[T]): Boolean = {
    require(x.length == y.length)
    val listBool = x.indices.map(i => {x(i)==y(i)})
    listBool.forall(identity)
  }

  def allEqualDouble(x: List[Double], y:List[Double], tol:Double = 0D): Boolean = {
    require(x.length == y.length)
    val listBool = x.indices.map(i => {abs(x(i) - y(i)) <= tol})
    listBool.forall(identity)
  }

  def getCondBlockPartition(rowPartition: Array[Array[Int]], colPartition: Array[Int]): Array[(Int, Int)] = {
    val blockPartitionMat = colPartition.par.map(l => {
      DenseMatrix.tabulate[(Int, Int)](rowPartition.head.length, 1) {
        (i, _) => (rowPartition(l)(i), l)
      }}).reduce(DenseMatrix.horzcat(_,_))
    blockPartitionMat.t.toArray
  }

  def getBlockPartition(rowPartition: List[Int], colPartition: List[Int]): List[Int] = {
    val n = rowPartition.length
    val p = colPartition.length
    val blockBiPartition: List[(Int, Int)] = DenseMatrix.tabulate[(Int, Int)](n,p)(
      (i, j) => (rowPartition(i), colPartition(j))).toArray.toList
    val mapBlockBiIndexToBlockNum = blockBiPartition.distinct.zipWithIndex.toMap
    blockBiPartition.map(mapBlockBiIndexToBlockNum(_))
  }

  def combineRedundantAndCorrelatedColPartitions(redundantColPartition: Array[Int],
                                                 correlatedColPartitions: Array[Array[Int]]): Array[Int] = {
    val reducedCorrelatedPartition = correlatedColPartitions.reduce(_++_)
    val orderedRedundantColPartition = redundantColPartition.zipWithIndex.sortBy(_._1)

    relabel(orderedRedundantColPartition.indices.map(j => {
      val vj  = orderedRedundantColPartition(j)._1
      val idx = orderedRedundantColPartition(j)._2
      val wj  = reducedCorrelatedPartition(j)
      (idx, vj, wj)
    }).sortBy(_._1).map(c => (c._2, c._3)).toArray)
  }

  def getMCCBlockPartition(redundantColPartition: Array[Int],
                           correlatedColPartitions: Array[Array[Int]],
                           rowPartitions: Array[Array[Int]]) : Array[Int] = {

    println("getMCCBlockPartition")

    val combinedColPartition = combineRedundantAndCorrelatedColPartitions(redundantColPartition, correlatedColPartitions)

    val rowPartitionDuplicatedPerColCluster = correlatedColPartitions.indices.map(h => {
      Array.fill(correlatedColPartitions(h).distinct.length)(rowPartitions(h))
    }).reduce(_++_)

    val indexBiClusters: Array[(Int, Int)] = getCondBlockPartition(rowPartitionDuplicatedPerColCluster, combinedColPartition)

    relabel(indexBiClusters)

  }


  def updateColPartition(formerColPartition: List[Int],
                         colToUpdate: Int,
                         newColPartition: List[Int]): List[Int]={
    var newGlobalColPartition = formerColPartition
    val otherThanlMap:Map[Int,Int] = formerColPartition.filter(_!=colToUpdate).distinct.sorted.zipWithIndex.toMap
    val L = max(formerColPartition)

    var iterNewColPartition = 0
    for( j <- newGlobalColPartition.indices){
      if(formerColPartition(j)==colToUpdate){
        newGlobalColPartition = newGlobalColPartition.updated(j,newColPartition(iterNewColPartition)+L)
        iterNewColPartition +=1
      } else {
        newGlobalColPartition = newGlobalColPartition.updated(j,otherThanlMap(formerColPartition(j)))
      }
    }
    newGlobalColPartition
  }

  def generateCombinationWithReplacement(maxK: Int, L: Int): List[List[Int]] ={
    List.fill(L)((1 to maxK).toList).flatten.combinations(L).toList
  }

  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def getRowPartitionFromDataWithRowPartition(precData: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[List[Int]] = {
    precData.map(row => (row._1, row._3))
      .collect().sortBy(_._1).map(_._2).toList.transpose
  }

  def getSizeAndSumByBlock(data: RDD[((Int, Int), Array[DenseVector[Double]])]): RDD[((Int, Int), (DenseVector[Double], Int))] = {
    data
      .map(r => (r._1, (r._2.reduce(_+_), r._2.length)))
      .reduceByKey((a,b) => (a._1 + b._1, a._2+b._2))
  }

  private def getDataByBlock(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             partitionPerColBc: Broadcast[DenseVector[Int]],
                             KVec: List[Int]): RDD[((Int, Int), Array[DenseVector[Double]])] = {
    data.flatMap(row => {
      KVec.indices.map(l => {
        val rowDv = DenseVector(row._2)
        val rowInColumn = rowDv(partitionPerColBc.value:==l)
        ((l, row._3(l)), rowInColumn.toArray)
      })
    }).cache()

  }

  private def getCovarianceMatrices(dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])],
                                    meanByBlock: Map[(Int, Int), DenseVector[Double]],
                                    sizeBlockMap: Map[(Int, Int), Int],
                                    KVec: List[Int],
                                    fullCovariance: Boolean=true): immutable.IndexedSeq[immutable.IndexedSeq[DenseMatrix[Double]]] = {
    val covFunction = unitCovFunc(fullCovariance)

    val sumCentered = dataPerColumnAndRow.map(r => {
      (r._1, r._2.map(v => covFunction(v - meanByBlock(r._1))).reduce(_+_))
    }).reduceByKey(_+_).collect().toMap
    KVec.indices.map(l => {
      (0 until KVec(l)).map(k => {
        sumCentered(l,k)/(sizeBlockMap(l,k).toDouble-1)
      })
    })
  }

  def getMeansAndCovariances(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             colPartition: Broadcast[DenseVector[Int]],
                             KVec: List[Int],
                             fullCovariance: Boolean): (List[List[DenseVector[Double]]], immutable.IndexedSeq[immutable.IndexedSeq[DenseMatrix[Double]]], RDD[((Int, Int), Int)], Map[(Int, Int), Int]) = {
    val dataByBlock: RDD[((Int, Int), Array[DenseVector[Double]])] = getDataByBlock(data, colPartition, KVec)
    val sizeAndSumBlock = getSizeAndSumByBlock(dataByBlock)
    val sizeBlock = sizeAndSumBlock.map(r => (r._1, r._2._2))
    val sizeBlockMap = sizeBlock.collect().toMap
    val meanByBlock: Map[(Int, Int), DenseVector[Double]] = sizeAndSumBlock.map(r => (r._1, r._2._1 / r._2._2.toDouble)).collect().toMap
    val covMat = getCovarianceMatrices(dataByBlock, meanByBlock, sizeBlockMap, KVec, fullCovariance)
    val listMeans = KVec.indices.map(l => {
      (0 until KVec(l)).map(k => {meanByBlock(l, k)}).toList
    }).toList
    (listMeans, covMat, sizeBlock, sizeBlockMap)
  }

  def getLoadings(covarianceMatrix: DenseMatrix[Double],
                  nMaxEigenValues: Int= 3,
                  thresholdVarExplained: Double=0.99): DenseMatrix[Double] = {

    val (sortedEigenVectors, sortedEigenValues) = getLoadingsAndEigenValues(covarianceMatrix)

    val normalizedEigenValues = sortedEigenValues/sum(sortedEigenValues)
    val cumulatedVarExplained = normalizedEigenValues.toArray.map{var s = 0D; d => {s += d; s}}
    val idxMaxVarianceExplained = cumulatedVarExplained.toList.indexWhere(_> thresholdVarExplained)
    val idxKept = min(idxMaxVarianceExplained, nMaxEigenValues)
    val keptEigenVectors = sortedEigenVectors(::, 0 until idxKept)
    keptEigenVectors.t
  }

  def getLoadingsAndVarianceExplained(covarianceMatrix: DenseMatrix[Double],
                                      nMaxEigenValues: Int= 3,
                                      thresholdVarExplained: Double=0.999999): (DenseMatrix[Double], DenseVector[Double]) = {

    require(thresholdVarExplained >= 0 & thresholdVarExplained < 1, "thresholdVarExplained should be >= 0 and <= 1")

    val (sortedEigenVectors, sortedEigenValues) = getLoadingsAndEigenValues(covarianceMatrix)

    val normalizedEigenValues = sortedEigenValues/sum(sortedEigenValues)
    val cumulatedVarExplained = normalizedEigenValues.toArray.map{var s = 0D; d => {s += d; s}}
    val idxMaxVarianceExplained = cumulatedVarExplained.toList.indexWhere(_> thresholdVarExplained)
    val idxKept = min(idxMaxVarianceExplained+1, nMaxEigenValues)
    val keptEigenVectors = sortedEigenVectors(::, 0 until idxKept)

    (keptEigenVectors.t, normalizedEigenValues.slice(0, idxKept))
  }

  def getLoadingsAndEigenValues(covarianceMatrix: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {

    val EigSym(eVal, eVec) = eigSym(covarianceMatrix)
    val sortedEigVal = DenseVector(eVal.toArray.sorted.reverse)
    val sortedEigVec = DenseMatrix((0 until eVec.rows).map(i => (eVec(::,i), eVal(i))).sortBy(-_._2).map(_._1):_*)
    (sortedEigVec.t, sortedEigVal)
  }


  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double]): DenseMatrix[Double] = {

    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)

    val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
    val XMatCentered: DenseMatrix[Double] = XMat - modeMat
    val covmat = (XMatCentered.t * XMatCentered)/ (X.length.toDouble-1)

    round(covmat,8)
  }

  def mean(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def factorial(n: Double): Double = {
    if (n == 0) {1} else {n * factorial(n-1)}
  }

  def logFactorial(n: Double): Double = {
    if (n == 0) {0} else {log(n) + logFactorial(n-1)}
  }

  def round(m: DenseMatrix[Double], digits:Int): DenseMatrix[Double] = {
    m.map(round(_,digits))
  }

  def round(x: Double, digits:Int): Double = {
    val factor: Double = Math.pow(10,digits)
    Math.round(x*factor)/factor
  }

  def matrixToDataByCol(data: DenseMatrix[DenseVector[Double]]): Array[Array[DenseVector[Double]]] = {
    data(::,*).map(_.toArray).t.toArray
  }

  def addDuplicate(dm: DenseMatrix[DenseVector[Double]], idx: Int, nDuplicate: Int) : DenseMatrix[DenseVector[Double]] = {
    val dataList = matrixToDataByCol(dm)
    val dataWithDuplicate: Array[Array[DenseVector[Double]]] = insertList(dataList, idx, Array.fill(nDuplicate-1)(dataList(idx))).toArray
    DenseMatrix( dataWithDuplicate:_*).t
  }


  def getScores(estimatedRowPartition: Array[Int], trueRowPartition: Array[Int]): (Double, Double, Double, Int) = {
    (
      adjustedRandIndex(estimatedRowPartition, trueRowPartition),
      randIndex(estimatedRowPartition, trueRowPartition),
      NormalizedMutualInformation.sum(estimatedRowPartition, trueRowPartition),
      estimatedRowPartition.distinct.length
    )
  }

  def getScores(estimatedRowPartition: Array[Int], trueRowPartition: Array[Int],
                estimatedColPartition: Array[Int], trueColPartition: Array[Int]): (Double, Double, Double, Double, Int) = {
    (
      adjustedRandIndex(estimatedRowPartition, trueRowPartition),
      adjustedRandIndex(estimatedColPartition, trueColPartition),
      randIndex(estimatedRowPartition, trueRowPartition),
      NormalizedMutualInformation.sum(estimatedRowPartition, trueRowPartition),
      estimatedRowPartition.distinct.length
    )
  }

  def distanceMatrix[T](data: List[T], distance: (T, T) => Double): DenseMatrix[Double] = {
    val DTri = DenseMatrix.zeros[Double](data.length, data.length)
    data.indices.par.foreach(i => {
      (0 until i).par.foreach(j => {
        DTri.update(i, j, distance(data(i), data(j)))
      })
    })
    DTri + DTri.t
  }

  def euclidDistanceVec(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    sqrt(sum(breeze.numerics.pow(x - y, 2)))
  }


  def distMax(x: List[DenseVector[Double]], y:List[DenseVector[Double]]): Double = {
    x.indices.map(i => {
      euclidDistanceVec(x(i), y(i))
    }).max
  }

  def distMin(x: List[DenseVector[Double]], y:List[DenseVector[Double]]): Double = {
    x.indices.map(i => {
      euclidDistanceVec(x(i), y(i))
    }).min
  }

  def distMean(x: List[DenseVector[Double]], y:List[DenseVector[Double]]): Double = {
    x.indices.map(i => {
      euclidDistanceVec(x(i), y(i))
    }).sum / x.length.toDouble
  }
}
