from libc.math cimport sqrt


cpdef double _single_dist_merge(double dist_xz, double dist_yz, double dist_xy, int size_z, int size_x, int size_y):
    return min(dist_xz, dist_yz)


cpdef double _complete_dist_merge(double dist_xz, double dist_yz, double dist_xy, int size_z, int size_x, int size_y):
    return max(dist_xz, dist_yz)


cpdef double _average_dist_merge(double dist_xz, double dist_yz, double dist_xy, int size_z, int size_x, int size_y):
    return (size_x * dist_xz + size_y * dist_yz) / (size_x + size_y)


cpdef double _ward_dist_merge(double dist_xz, double dist_yz, double dist_xy, int size_z, int size_x, int size_y):
    cdef double t = 1.0 / (size_x + size_y + size_z)

    return sqrt((size_z + size_x) * t * dist_xz * dist_xz +
                (size_z + size_y) * t * dist_yz * dist_yz -
                size_z * t * dist_xy * dist_xy)
