
import numpy as np

FILENAME = r'datasets/movement_libras.data'
D = 2
training_set_percentage = 0.7
validation_set_percentage = 0.1
test_set_percentage = 0.2
num_labels = 15
min_label = 1
max_label = min_label + num_labels - 1

PERMUTATIONS = np.array([121, 247, 192, 230, 254, 298,  98,   5, 130,  83, 308,  20, 332,
                         349, 324,  88, 169, 289, 302, 226, 269, 293, 103, 255, 102,  19,
                          67, 143, 314,  44, 268,  46, 215, 117, 322, 212,  85, 356, 159,
                           1, 141, 224, 131,  33, 250,  39, 218, 318, 186, 167,  70, 180,
                         348, 248, 197, 239, 183,  73,  53, 189, 174, 139,   3,  71, 312,
                         311, 145,  35, 168,  43, 129, 246, 191, 245,  28, 249,  84, 272,
                         229, 100, 251, 307, 256, 347, 188, 336,  23, 240, 257,  48, 294,
                         195,  37, 177,  15, 271, 161, 171, 134, 223,  12, 301,   6, 203,
                         323,  94, 136, 149, 228, 306,  10, 110, 146, 214,  31, 154,   8,
                         303, 172, 242,   9, 114,  97, 295, 150,  68,  58,  52,  60, 153,
                         335, 258, 152, 346, 266, 309, 196, 207, 127, 142, 199, 112,  51,
                          79, 280, 282, 225, 317,  69,  30,  65,  13,  57, 160, 132, 194,
                          18, 231, 305, 198, 267, 219, 241, 140, 101,  96, 259, 173,  40,
                         162, 222, 227, 166, 148, 209, 297, 253, 328, 333, 210, 252,  90,
                         358, 178, 200,  64, 126, 279, 190, 105, 274, 315, 135,  49, 221,
                         329,  99,  47,  95, 176, 124, 217, 352, 108, 185, 326, 353, 106,
                         264, 156,  16,  56, 115, 120, 182, 123, 283,  63,  14, 116, 345,
                         327,  62, 337, 277, 235, 320, 325, 181,   2,  38, 170, 175, 344,
                         275, 265, 237, 319, 299, 296,  24, 321,  92, 107,  61,  59,  11,
                          36, 165, 118, 138, 147, 109, 144, 179, 133, 163, 304, 158, 137,
                         300, 202,  72, 211, 287, 193, 243, 205,  74, 331,  29,  75, 104,
                          21, 262,   7, 288, 244, 213, 155, 128, 260,  25, 157,  86,  80,
                         111,   0, 334, 281,  22, 201, 316, 290,  78,  17, 359, 340,  32,
                         184,  34, 313,  76,  89, 273, 206,  41,  26,   4, 292,  66, 330,
                          81, 278, 263, 286, 119, 351, 310,  87,  82,  55, 122,  45, 285,
                         350,  50, 164, 357,  91, 270, 284, 261, 291, 187,  54, 238, 339,
                         338, 204, 342, 343,  93, 220, 354,  27, 151,  77, 216, 125, 113,
                         276, 233, 208,  42, 355, 341, 236, 232, 234])

assert training_set_percentage + \
        validation_set_percentage + \
        test_set_percentage == 1.0

class Dataset(object):
    def __init__(self):
        with open(FILENAME, 'r') as f:
            lines = f.readlines()

        N = len(lines)
        T = (len(lines[0].split(','))-1) / D

        # Check that T*D == number of measurements in sample
        assert T*D == (len(lines[0].split(','))-1)

        #  samples = np.array(N, T, D, 1)
        data_from_csv = np.genfromtxt(FILENAME, delimiter=',')

        labels = data_from_csv[:,-1]
        labels_one_hot = \
                (np.arange(min_label, max_label + 1) == \
                labels[:,None]).astype(np.float32)

        assert labels_one_hot[359, 14] == 1

        labels_one_hot = labels_one_hot[PERMUTATIONS,:]

        samples = data_from_csv[:,:-1]
        samples_x = samples[:,0::D]
        samples_y = samples[:,1::D]
        dataset = np.stack((samples_x, samples_y), axis=-1)
        dataset = dataset[:,:,:,None]

        assert (N, T, D, 1) == dataset.shape
        assert dataset[0,0,0] == 0.79691
        assert dataset[1,0,1] == 0.27315
        assert dataset[N-1,T-1,1] == 0.49306

        dataset = dataset[PERMUTATIONS,:,:,:]

        training_set_len = int(round(training_set_percentage*N))
        training_set = dataset[:training_set_len, :, :, :]
        training_labels = labels_one_hot[:training_set_len, :]

        cur_place = training_set_len
        validation_set_len = int(round(validation_set_percentage*N))
        validation_set = dataset[cur_place:(cur_place + validation_set_len), :, :, :]
        validation_labels = labels_one_hot[cur_place:(cur_place + validation_set_len), :]

        cur_place = training_set_len + validation_set_len
        test_set_len = int(round(test_set_percentage*N))
        test_set = dataset[cur_place:(cur_place + test_set_len), :, :, :]
        test_labels = labels_one_hot[cur_place:(cur_place + test_set_len), :]

        assert training_set_len + validation_set_len + test_set_len == N

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.training_labels = training_labels
        self.validation_labels = validation_labels
        self.test_labels = test_labels
        self.N = N
        self.T = T
        self.D = D

    def get_training_set(self):
        return self.training_set.astype(np.float32)

    def get_validation_set(self):
        return self.validation_set.astype(np.float32)

    def get_test_set(self):
        return self.test_set.astype(np.float32)

    def get_dimensions(self):
        return (self.N, self.T, self.D)

    def get_num_of_labels(self):
        return num_labels

    def get_train_labels(self):
        return self.training_labels

    def get_validation_labels(self):
        return self.validation_labels

    def get_test_labels(self):
        return self.test_labels

