import flax
import jax

class SimpleCNN(flax.nn.Module):
	def apply(self, x, 
		train = None, 
		channels = [32,64,64,128], 
		classes = 10):
		#the train doesn't do anything since no batchnorm

		for features in channels:
			x = flax.nn.Conv(x, features=features, kernel_size=(3, 3))
			x = flax.nn.relu(x)
			x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

		#getting it flat
		x = x.reshape((x.shape[0], -1))

		# final two layers
		# x = flax.nn.Dense(x, features=128)
		# x = flax.nn.relu(x)
		x = flax.nn.Dense(x, features=classes)
		x = flax.nn.log_softmax(x)
		return x
	
class KerasResNets(flax.nn.Module): 
# Keras examples based ResNet
# the default settings are ResNet20v1
# should get >91.5% on CIFAR-10 test
# after 200 epochs of training
  
  def apply(self, x, num_classes =10, use_batch_norm = True, train=True):

    depth=(3*6+2)

    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation="relu",
                     conv_first=True,
                     train=True,
                     ):
      
      if use_batch_norm:
        batch_norm = flax.nn.BatchNorm.partial(use_running_average=not train,momentum=0.9, epsilon=1e-5)
      
      if activation == "relu":
        f = flax.nn.relu
      elif activation is None:
        f = lambda q: q

      a = inputs
      if conv_first:
        a = flax.nn.Conv(a, features=num_filters,strides=(strides,strides),padding="SAME",kernel_size=(kernel_size, kernel_size), bias=False)
        if use_batch_norm:
          a = batch_norm(a)
        a = f(a)
      else:
        if use_batch_norm:
          a = batch_norm(a)
        a = f(a)
        a = flax.nn.Conv(a, features=num_filters,strides=(strides,strides),padding="SAME",kernel_size=(kernel_size, kernel_size), bias=False)
      
      return a

    if (depth - 2) % 6 != 0:
          raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
      # Start model definition.
      
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = x
    x = resnet_layer(inputs=inputs, train=train)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                            num_filters=num_filters,
                            strides=strides, train=train)
            
            y = resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None, train=train)
            
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                kernel_size=1,
                                strides=strides,
                                activation=None, train=train)
                
            x = x+y
            x = flax.nn.relu(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU

    x = flax.nn.avg_pool(x, window_shape=(7, 7), strides=(7, 7))

    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, features=num_classes)
    x = flax.nn.log_softmax(x)

    return x

class WideResnetBlock(flax.nn.Module):
  """Defines a single wide ResNet block."""

  def apply(self, x, channels, strides=(1, 1), dropout_rate=0.0, train=True):
    batch_norm = flax.nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)

    y = batch_norm(x, name='bn1')
    y = jax.nn.relu(y)
    y = flax.nn.Conv(y, channels, (3, 3), strides, padding='SAME', name='conv1')
    y = batch_norm(y, name='bn2')
    y = jax.nn.relu(y)
    if dropout_rate > 0.0:
      y = flax.nn.dropout(y, dropout_rate, deterministic=not train)
    y = flax.nn.Conv(y, channels, (3, 3), padding='SAME', name='conv2')

    # Apply an up projection in case of channel mismatch
    if (x.shape[-1] != channels) or strides != (1, 1):
      x = flax.nn.Conv(x, channels, (3, 3), strides, padding='SAME')
    return x + y


class WideResnetGroup(flax.nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            dropout_rate=0.0,
            train=True):
    for i in range(blocks_per_group):
      x = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          dropout_rate,
          train=train)
    return x


class WideResnet(flax.nn.Module):
  """Defines the WideResnet Model."""

  def apply(self,
            x,
            blocks_per_group,
            channel_multiplier,
            num_outputs,
            dropout_rate=0.0,
            train=True):

    x = flax.nn.Conv(
        x, 16, (3, 3), padding='SAME', name='init_conv')
    x = WideResnetGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        dropout_rate=dropout_rate,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        train=train)
    x = flax.nn.BatchNorm(
        x,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5)
    x = jax.nn.relu(x)
    x = flax.nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, num_outputs)

    x = flax.nn.log_softmax(x)

    return x
