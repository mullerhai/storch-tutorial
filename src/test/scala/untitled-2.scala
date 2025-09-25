//import ai.onnxruntime.*
//import org.pytorch.*
//
//import scala.collection.mutable.ArrayBuffer
//import scala.jdk.CollectionConverters.*
//
//object TorchScriptToOnnxPrediction {
//  def main(args: Array[String]): Unit = {
//    // 加载 TorchScript 模型
//    val torchScriptModelPath = "path/to/your/torchscript_model.pt"
//    val module = torch.jit.load(torchScriptModelPath)
//
//    // 定义输入示例
//    val inputTensor = torch.rand(Array(1, 3, 224, 224))
//    val inputNames = Array("input")
//    val outputNames = Array("output")
//    val dynamicAxes = Map(
//      "input" -> Map(0 -> "batch_size"),
//      "output" -> Map(0 -> "batch_size")
//    ).asJava
//
//    // 将 TorchScript 模型转换为 ONNX 模型
//    val onnxModelPath = "path/to/your/converted_model.onnx"
//    torch.onnx.export(
//      module,
//      inputTensor,
//      onnxModelPath,
//      inputNames,
//      outputNames,
//      dynamicAxes,
//      opsetVersion = 11
//    )
//
//    // 加载 ONNX 模型
//    val sessionOptions = new OrtSession.SessionOptions()
//    val session = OrtEnvironment.getEnvironment.createSession(onnxModelPath, sessionOptions)
//
//    // 准备输入数据
//    val onnxInputTensor = createInputTensor()
//
//    // 运行预测
//    val inputs = Map[String, OnnxTensor](session.getInputNames.asScala.next() -> onnxInputTensor).asJava
//    val results = session.run(inputs)
//
//    // 处理输出结果
//    val outputTensor = results.get(0).asInstanceOf[OnnxTensor]
//    val outputData = outputTensor.getFloatBuffer.array()
//    println(s"预测结果: ${outputData.mkString(", ")}")
//
//    // 释放资源
//    results.close()
//    onnxInputTensor.close()
//    session.close()
//  }
//
//  private def createInputTensor(): OnnxTensor = {
//    // 这里需要根据你的模型输入要求修改输入数据
//    val inputData = Array.fill[Float](3 * 224 * 224)(1.0f)
//    val shape = Array[Long](1, 3, 224, 224)
//    OnnxTensor.createTensor(OrtEnvironment.getEnvironment, inputData, shape)
//  }
//}
