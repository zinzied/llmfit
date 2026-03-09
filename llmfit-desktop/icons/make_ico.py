import struct
# 22 bytes header, 40 bytes BMP header, 4 bytes pixel data (32bpp)
ico_data = b'\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x20\x00\x2c\x00\x00\x00\x16\x00\x00\x00'
bmp_header = struct.pack('<IiiHHIIIIII', 40, 1, 2, 1, 32, 0, 4, 0, 0, 0, 0)
pixels = b'\xff\xff\xff\xff'
with open('icon.ico', 'wb') as f:
    f.write(ico_data + bmp_header + pixels)
