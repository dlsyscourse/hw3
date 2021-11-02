"""NDDArray backed computation backend.

This backend uses cuda backend_ndarray for cached data and computation.
"""
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr


class NDDevice(Device):
    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=self.nd_device)

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=self.nd_device)

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def enabled(self):
        return self.nd_device.enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)



class CUDADevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.cuda()
        self.device_id = device_id

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __str__(self):
        return self.__repr__()


class CPUDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        self.nd_device = nd.cpu()
        self.device_id = device_id

    def __repr__(self):
        return "cpu(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CPU, self.device_id)

    def __str__(self):
        return self.__repr__()



def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def cpu() -> CPUDevice:
    return CPUDevice()


def register_nd_compute(name, value=None):
    """Register the compute property based on backend_ndarray
    nd computation can be shared across multiple backends.
    """
    return register_op_attr(name, "nd_compute", value)


# device specific computations
@register_nd_compute("EWiseAdd")
def add(inputs, attrs):
    return inputs[0] + inputs[1]


@register_nd_compute("AddScalar")
def add_scalar(inputs, attrs):
    return inputs[0] + attrs["scalar"]


@register_nd_compute("EWiseMul")
def mul(inputs, attrs):
    return inputs[0] * inputs[1]


@register_nd_compute("MulScalar")
def mul(inputs, attrs):
    return inputs[0] * attrs["scalar"]
