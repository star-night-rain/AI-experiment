import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time


def get_args(image):
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--image', default=image)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--selection_rate', type=float, default=0.4)
    parser.add_argument('--cross_rate', type=float, default=0.3)
    parser.add_argument('--variation_rate', type=float, default=0.05)
    args = parser.parse_args()
    return args


def transformation(image_arg, threshold):
    print(f'threshold:{threshold}')
    image_arg[image_arg > threshold] = 255
    image_arg[image_arg <= threshold] = 0
    # display image
    plt.imshow(image_arg, cmap='gray')
    plt.axis('off')
    plt.show()


def OTSU(image, threshold):
    size = image.shape[0] * image.shape[1]
    sum = np.sum(image)

    w0 = np.sum(image <= threshold)
    w1 = size - w0

    s0 = np.sum(image[image <= threshold])
    s1 = sum - s0

    u0 = s0 / w0 if w0 != 0 else 0
    u1 = s1 / w1 if w1 != 0 else 0

    w0 = w0 / size
    w1 = w1 / size

    g = w0 * w1 * pow(u0 - u1, 2)
    return g


class GA:

    def __init__(self, args):
        self.image = args.image
        self.M = args.M
        self.length = args.length
        self.selection_rate = args.selection_rate
        self.cross_rate = args.cross_rate
        self.variation_rate = args.variation_rate

        self.species = np.random.randint(0, 256, self.M)

    def selection(self):
        fitness = list()
        for species in self.species:
            fitness.append((OTSU(self.image, species), species))
        fitness.sort(reverse=True)
        parents = list()
        for _, species in fitness[:self.M]:
            parents.append(species)
        for _, species in fitness[self.M:]:
            if np.random.random() <= self.selection_rate:
                parents.append(species)
        return parents

    def cross(self, parents):
        children = list()
        for i in range(0, len(parents) - 1, 2):
            if np.random.random() <= self.cross_rate:
                position = np.random.randint(0, self.length)
                parent1 = format(parents[i], '08b')
                parent2 = format(parents[i + 1], '08b')
                child1 = int(parent1[position:] + parent2[:position], 2)
                child2 = int(parent2[position:] + parent1[:position], 2)
                children.append(child1)
                children.append(child2)
        return children

    def variation(self, children):
        species = list()
        for child in children:
            if np.random.random() <= self.variation_rate:
                position = np.random.randint(0, self.length)
                child = child ^ (1 << position)
            species.append(child)
        return species

    def evolution(self):
        parents = self.selection()
        children = self.cross(parents)
        children = self.variation(children)
        species = parents + children
        self.species = species

    def get_threshold(self):
        fitness = list()
        for species in self.species:
            fitness.append((OTSU(self.image, species), species))
        fitness.sort(reverse=True)
        return fitness[0][1]


def main():
    image_name = '1.jpg'
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    args = get_args(image)
    ga = GA(args)

    start_time = time.perf_counter()
    for epoch in range(args.epoch):
        ga.evolution()
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f'execution time:{duration:.4f}s')

    threshold = ga.get_threshold()
    transformation(image, threshold)


if __name__ == '__main__':
    main()
