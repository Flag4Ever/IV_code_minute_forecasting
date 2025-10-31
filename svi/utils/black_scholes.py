def normalizing_transforms(r, z, iv):
    a = - z / iv
    total_volatility = iv * r
    d1, d2 = a + total_volatility / 2, a - total_volatility / 2
    return d1, d2
