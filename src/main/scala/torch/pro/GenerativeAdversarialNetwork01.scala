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

class GenerativeAdversarialNetwork[D <: FloatNN : Default](image_size: Int = 784, h_dim: Int = 400, z_dim: Int = 20) extends HasParams[D] {

  //  F.maxPool2d()
}

object GenerativeAdversarialNetwork01 {
  val latent_size = 64
  val hidden_size = 256
  val image_size = 784
  val num_epochs = 200
  val batch_size = 100
  val sample_dir = "samples"
  val D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

//  #Generator
  val G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
}
