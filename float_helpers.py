import struct

float2int = lambda f: struct.unpack('i', struct.pack('f', f))[0]
float2hex = lambda f: hex(float2int(f))
MASK_23B = (1 << 23) - 1
BIAS = 127


def test_one_to_hex():
    assert float2hex(1.0) == '0x3f800000'


def float2parts(f):
    x = float2int(f)
    s = x >> 31
    e = x >> 23 & 0xff
    f = x & MASK_23B
    return (s, e, f)


def test_floatparts_one():
    assert float2parts(1.0) == (0, BIAS, 0)


def parts2float(t):
    s, e, f = t
    assert s & 1 == s
    assert e & 0xff == e
    assert f & MASK_23B == f
    x = s << 31 | e << 23 | f
    return struct.unpack('f', struct.pack('i', x))[0]


def test_parts2float():
    for f in (1.0, 1.5, 2.0, 3.75, 0.0625):
        assert parts2float(float2parts(f)) == f, f
