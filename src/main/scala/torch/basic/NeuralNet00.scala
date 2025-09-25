package torch
package basic

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.nn.functional as F
import torch.optim.Adam
import torch.utils.data.dataset.custom.FashionMNIST
import java.nio.file.Paths
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.HasParams
import torch.{Default, FloatNN, *}
import scala.util.{Random, Using}
class NeuralNet[D <: FloatNN : Default](input_size: Int = 784, 
                                        hidden_size: Int = 500, 
                                        num_classes: Int = 10) extends HasParams[D] {
  val fc1 = register(nn.Linear(input_size, hidden_size))
  val relu = register(nn.ReLU())
  val fc2 = register(nn.Linear(hidden_size, num_classes))

  def apply(input: Tensor[D]): Tensor[D] = {
    var out = fc1(input)
    out = relu(out)
    out = fc2(out)
    out
  }
}

object NeuralNet00 {

  def main(): Unit = {
    val model = NeuralNet[Float32]()
    val criterion = nn.loss.CrossEntropyLoss()
    val optimizer = torch.optim.Adam(model.parameters, lr = 0.00001)
  }
}
