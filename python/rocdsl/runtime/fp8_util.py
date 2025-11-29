import struct

def to_byte(v):
    # E4M3FNUZ
    # 1 sign, 4 exponent, 3 mantissa
    # bias 8
    # no infinity, no NaN (NaN is 0x80)
    # 0 is 0x00
    
    if v == 0.0:
        return 0x00
    
    # Handle sign
    sign = 0
    if v < 0:
        sign = 0x80
        v = -v
        
    # Get float bits
    packed = struct.pack('>f', v)
    i = struct.unpack('>I', packed)[0]
    
    f_exp = (i >> 23) & 0xFF
    f_mant = i & 0x7FFFFF
    
    # Unbias float32 exponent
    exp = f_exp - 127
    
    # Rebias for E4M3FNUZ (bias 8)
    new_exp = exp + 8
    
    if new_exp > 15: # Overflow
        return sign | 0x7F
        
    if new_exp <= 0: # Denormal
         # Denormalize: 1.mant * 2^exp = 0.mant' * 2^0
         # We have 23 bits mantissa + implicit 1
         # Shift to align with 3 bit mantissa
         # 1.xxxxx... * 2^exp
         # We want format 0.yyy * 2^0
         # shift amount = 1 - new_exp
         
         # Full mantissa with implicit 1
         full_mant = f_mant | 0x800000
         
         # We want to end up with 3 bits.
         # Currently 24 bits (bit 23 is implicit 1)
         # If exp is 0, we shift right by 1 to get 0.1xxxx
         # If exp is -1, we shift right by 2 to get 0.01xxx
         
         shift = 1 - new_exp
         # We want the result to be in the lower 3 bits? No, the whole thing is the mantissa for denormals (exp=0)
         # For E4M3FNUZ, denormals have exp=0.
         
         # Let's align to the position of the 3 mantissa bits (bits 0-2)
         # The implicit 1 is at bit 23 in float32.
         # We want it to move to bit 2 (or lower)
         # shift = 23 - 2 + shift_due_to_exp = 21 + (1 - new_exp) = 22 - new_exp
         
         val = full_mant >> (22 - new_exp)
         return sign | (val & 0x7)

    # Normal case
    # 4 bits exponent, 3 bits mantissa
    new_mant = (f_mant >> 20) & 0x7
    return sign | (new_exp << 3) | new_mant
