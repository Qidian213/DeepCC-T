import sys
import numpy as np
import numbers
import tensorflow as tf

def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]

#fe = [-7.95455635e-01, -6.85662210e-01 ,-6.33207083e-01, -1.40617239e+00,
#   1.65666854e+00,  7.78477490e-01,  4.13639218e-01,  1.56610298e+00,
#   1.03642896e-01, -6.26503170e-01, -1.63301453e-01, -3.55674833e-01,
#  -2.69647211e-01,  9.11448777e-01, -1.02938724e+00, -1.19604744e-01,
#  -7.32812472e-03,  8.28335166e-01, -4.53803129e-02,  3.50463152e-01,
#   1.49852836e+00,  7.65579581e-01,  6.29868507e-01, -6.22135580e-01,
#  -2.42215335e-01, -1.08994162e+00, -8.77791703e-01,  9.89699841e-01,
#   9.90987122e-01,  6.78343624e-02, -6.68448210e-02,  6.04805887e-01,
#  -2.56032079e-01, -2.05195427e+00, -1.12726986e-01,  5.40941320e-02,
#   3.51185769e-01,  1.50325608e+00,  4.30240065e-01,  6.69089019e-01,
#   3.75487924e-01, -1.18156731e+00,  8.43675077e-01,  8.69958103e-01,
#   1.02698588e+00, -9.55180287e-01,  4.18605469e-02, -3.65568429e-01,
#   6.15232170e-01, -1.24067798e-01,  6.81436285e-02, -7.47346342e-01,
#   8.82668674e-01, -1.33605874e+00, -1.35415757e+00,  9.07526791e-01,
#  -1.43234968e-01, -7.86721647e-01,  1.79714218e-01, -6.08267307e-01,
#  -1.23202550e+00, -5.84441602e-01, -1.15516126e-01, -5.56537867e-01,
#  -5.68276286e-01, -6.67798042e-01,  6.41098619e-01, -9.19512331e-01,
#   3.02285522e-01, -1.53131232e-01,  1.07643318e+00, -4.59839433e-01,
#  -1.36854362e+00,  6.59813285e-01,  1.05665219e+00,  1.87346768e+00,
#   3.93893600e-01,  1.58856913e-01,  1.70471799e+00, -2.53166127e+00,
#   8.62807781e-02, -1.72109568e+00,  1.77071527e-01,  1.53216434e+00,
#   1.29825854e+00, -7.53714502e-01,  9.98314559e-01, -8.32642078e-01,
#  -7.69476499e-03,  1.13118017e+00, -4.07275885e-01,  9.59977269e-01,
#  -1.43860385e-01, -9.20496166e-01, -3.66329193e-01,  6.00143373e-01,
#   9.66092423e-02, -1.28283417e+00, -5.14184415e-01,  1.48718131e+00,
#   1.32180542e-01, -5.58032572e-01,  1.11029945e-01,  8.87744367e-01,
#   4.65452462e-01,  5.37216842e-01, -7.79009014e-02, -7.01277435e-01,
#   2.38400626e+00,  1.68282723e+00, -9.10158932e-01,  7.84041286e-01,
#   8.45085740e-01, -1.16154706e+00, -5.22633314e-01,  2.50917256e-01,
#  -1.83769032e-01,  5.51918209e-01, -4.75720853e-01,  4.46916193e-01,
#   6.23019695e-01, -8.48545492e-01, -2.75536329e-01, -5.52381933e-01,
#  -9.32174027e-01, -1.36525303e-01, -1.76950142e-01, -3.56358498e-01]

#a = [-0.84996873, -0.7107881,  -0.642646,   -1.334782,    1.6034037,   0.6782194,
#   0.45991054,  1.4638652,   0.23249736, -0.5960715,  -0.00539125, -0.47551188,
#  -0.30558133,  0.986086,   -0.89879316, -0.05905719, -0.14977221,  0.7056215,
#  -0.14697404,  0.36163846,  1.6004325,   0.82720244,  0.6236633,  -0.71632284,
#  -0.20596439, -1.0611264,  -0.9057392,   0.89307594,  1.1000005,   0.11354159,
#  -0.15852457,  0.4736619,  -0.08273888, -2.0895069,  -0.08996028, -0.04125955,
#   0.3148074,   1.5121036,   0.36367998,  0.622382,    0.42146426, -1.2901729,
#   0.8433884,   0.79443496,  0.9632506,  -0.8335845,   0.00472553, -0.54020476,
#   0.70845014, -0.14293317, -0.00889797, -0.46809852,  0.82675153, -1.2719295,
#  -1.224207,    0.875075,    0.00537344, -0.6835354,   0.12283206, -0.53991246,
#  -1.08925,    -0.5344788,  -0.1863719,  -0.48064706, -0.65018964, -0.67219126,
#   0.5689145,  -0.79918593,  0.35288242, -0.09169669,  0.84113914, -0.3734105,
#  -1.3029096,   0.8170331,   1.1020551,   1.927979,    0.45392722, -0.01049046,
#   1.691825,   -2.4599338,   0.19690631, -1.7088743,   0.07488932,  1.565341,
#   1.3506172,  -0.7670735,   1.0087773,  -0.9626112,  -0.13133365,  1.0539244,
#  -0.25376305,  1.0319284,  -0.06997913, -1.000552,   -0.32511315,  0.5898431,
#   0.08988629, -1.3078446,  -0.61202455,  1.4690497,   0.04805274, -0.5312492,
#   0.15056588,  0.8621065,   0.3962696,   0.58686626, -0.21445027, -0.5801945,
#   2.3139794,   1.656302,   -0.8945563,   0.8108471,   0.7554968,  -1.1221247,
#  -0.5560427,   0.25088152, -0.07756521,  0.5831301,  -0.5709566,   0.3809626,
#   0.6320324,  -0.93691427, -0.44507742, -0.63753045, -0.9165081,  -0.17597833,
#  -0.291259,   -0.26290408]
#b = [ 0.47303793,  0.31392825, -1.3174056,  -0.8098178,  -0.2536675,  -0.2650636,
#   0.672047,   -0.9488869,   1.3109587,  -0.12565082, -0.7476844 , -0.69744545,
#  -0.76319826,  0.29878905, -1.8607502,  -0.33403137,  0.2250745,   0.4141951,
#   0.5620803,   1.0638825,   1.9089311,  -0.27549857, -0.39491418, -0.58553,
#   0.40209708,  0.7945693,   0.2829329,   1.0809778,   1.0924639,   0.74223804,
#  -1.0139552,   1.7521629,   0.491319,   -0.11787625,  0.14074734,  1.1958553,
#   0.98035264, -0.6896183,   0.19133644,  1.6221774,   1.0873457,  -1.1025726,
#   0.20182875,  1.499408,    0.71731406, -0.60976326, -0.13848005,  0.91304135,
#  -0.27659413,  0.3024011,   0.8808978,   0.64513206, -0.9927348,   0.8355387,
#  -1.909827,    0.24038032, -1.1391776,  -1.1174145,   0.36268616,  1.2668371,
#  -0.7985614,   0.61525196, -0.44938856, -1.0973612,  -0.96432596, -1.2445998,
#  -1.6477115,   1.199412,   -0.5266097,   0.0043726,   0.01294904, -0.17730966,
#   0.2836082,  -1.9075187,  -1.7499981,   2.4282138,   0.01858903, -0.68745196,
#   0.1457975,   0.09887636, -0.7632798,   0.38893142, -0.20616905,  1.7302573,
#   0.6289535,  -2.0024204,  -1.4924773,  -0.07522956,  0.6981529 ,  0.661434,
#   0.20460965,  1.0107082,  -0.1453912,  -0.30343518, -0.35290635, -0.39662892,
#   1.0514568,   0.5811896,  -0.45851842, -0.9351927,  -0.50101775 ,-0.24040289,
#  -3.1814384,  -0.8488974,  -0.02256011,  0.73287874,  0.57859164 ,-1.0268437,
#   0.8758395,   0.77336705, -1.8485525,  -1.4834412,   0.9187003 , -1.5113713,
#  -0.07158987,  0.8530088,  -1.4149998,  -1.7384056,  -0.98108494, -0.63769364,
#   0.570032,   -0.49081063,  0.91019684, -1.8406003,  -0.4854365,   0.7084089,
#  -1.3861805,   0.43767825]

#vefe = np.array(fe)
#vea = np.array(a)
#veb = np.array(b)

#distfea = np.sqrt(np.sum(np.square(vefe-vea)))
#distfeb = np.sqrt(np.sum(np.square(vefe-veb)))

#print(distfea)
#print(distfeb)

#sum_q = 0
#for val in fe:
#    sum_q = sum_q+val**2
#print(sum_q)
#sum_a =0 
#for val in a:
#    sum_a = sum_a+val**2
#print(sum_a)
#sum_b =0
#for val in b:
#    sum_b = sum_b+val**2
#print(sum_b)

#distance = 0
#for val1, val2 in fe ,a:
#    distance = distance+(val1-val2)**2
#print(distance)

#distanc2 = 0
#for val1, val2 in fe ,b:
#    distanc2 = distanc2+(val1-val2)**2
#print(distanc2)









