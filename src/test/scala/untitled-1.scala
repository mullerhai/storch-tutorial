//import ai.onnxruntime.*
//
//import scala.collection.mutable.ArrayBuffer
//import scala.jdk.CollectionConverters.*
//
//object OnnxPredictionExample {
//  def main(args: Array[String]): Unit = {
//    // 加载 ONNX 模型
//    val sessionOptions = new OrtSession.SessionOptions()
//    val session = OrtEnvironment.getEnvironment.createSession("path/to/your/model.onnx", sessionOptions)
//
//    // 准备输入数据
//    val inputTensor = createInputTensor()
//
//    // 运行预测
//    val inputs = Map[String, OnnxTensor](session.getInputNames.asScala.next() -> inputTensor).asJava
//    val results = session.run(inputs)
//
//    // 处理输出结果
//    val outputTensor = results.get(0).asInstanceOf[OnnxTensor]
//    val outputData = outputTensor.getFloatBuffer.array()
//    println(s"预测结果: ${outputData.mkString(", ")}")
//
//    // 释放资源
//    results.close()
//    inputTensor.close()
//    session.close()
//  }
//
//  private def createInputTensor(): OnnxTensor = {
//    // 这里需要根据你的模型输入要求修改输入数据
//    val inputData = Array[Float](1.0f, 2.0f, 3.0f, 4.0f)
//    val shape = Array[Long](1, inputData.length)
//    OnnxTensor.createTensor(OrtEnvironment.getEnvironment, inputData, shape)
//  }
//}
