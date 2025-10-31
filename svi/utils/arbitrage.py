def butterfly(d1, d2, iv, div_dz, div_dzz):
    return (1 + div_dz * d1) * (1 + div_dz * d2) + iv * div_dzz


def calendar(r, x, iv, div_dr, div_dz):
    return ((iv - x * div_dz) / r + div_dr) / 2
