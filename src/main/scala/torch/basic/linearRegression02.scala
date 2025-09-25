package torch.basic

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
//import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam

object linearRegression02 {
  //  @main
  def main(): Unit =
    val input_size = 1
    val output_size = 1
    val num_epochs = 60
    val learning_rate = 0.001f

    val x_train = torch.Tensor(Seq(3.3, 4.4, 5.5, 6.71, 6.93, 4.169, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.312, 7.993, 3.1), requiresGrad = true).view(15, 1).to(dtype = torch.float32)
    val y_train = torch.Tensor(Seq(1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.68, 2.904, 1.3), requiresGrad = true).view(15, 1).to(dtype = torch.float32)

    println(x_train.shape)
    println(y_train.shape)
    val model = nn.Linear(input_size, output_size)
    val criterion = nn.loss.MSELoss()
    val optimizer = torch.optim.SGD(model.parameters(true), lr = learning_rate)
    //    (1 to 5).map(println)

    (1 to num_epochs).foreach(epoch => {
      val outputs = model(x_train)
      val loss = criterion(outputs, y_train)
      optimizer.zero_grad()
      loss.backward()
      println(s"loss grad_fn: ${loss.grad_fn()}")
      println(s"out grad_fn: ${outputs.grad_fn()}")
      optimizer.step()
      if (epoch + 1) % 5 == 0 then println(s"Epoch ${epoch} ,loss , ${loss.item}")
    })
}
