package torch
package pro

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.{FloatNN, *}
import torch.data.DataLoaderOptions
import torch.data.dataloader.*
import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
import torch.data.dataset.*
import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.conv.Conv2d

import scala.util.{Random, Using}

class VAEModel[D <: FloatNN : Default](image_size: Int = 784, h_dim: Int = 400, z_dim: Int = 20) extends HasParams[D] {

  val fc1 = register(nn.Linear(image_size, h_dim))
  val fc2 = register(nn.Linear(h_dim, z_dim))
  val fc3 = register(nn.Linear(h_dim, z_dim))
  val fc4 = register(nn.Linear(z_dim, h_dim))
  val fc5 = register(nn.Linear(h_dim, image_size))

  def apply(input: Tensor[D]): Tuple3[Tensor[D], Tensor[D], Tensor[D]] = {
    val Tuple2(mu, log_var) = encode(input)
    val z = reparameterize(mu, log_var)
    val x_reconst = decode(z)
    Tuple3(x_reconst, mu, log_var)
  }

  def encode(input: Tensor[D]): Tuple2[Tensor[D], Tensor[D]] = {
    val h = F.relu(fc1(input))
    Tuple2(fc2(h), fc3(h))
  }

  def reparameterize(mu: Tensor[D], log_var: Tensor[D]): Tensor[D] = {
    val std = torch.exp(log_var.div(2))
    val eps = torch.randn_like(std)
    mu.add(eps).add(std)
  }

  def decode(input: Tensor[D]): Tensor[D] = {
    val h = F.relu(fc4(input))
    F.sigmoid(fc5(h))
  }
}

object VAEModel {

}
