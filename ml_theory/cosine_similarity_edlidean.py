from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

pts1 = np.array([1, 2, 3])
pts2 = np.array([0, 5, 8])

# cosine similarity
print(np.linalg.norm(pts1) == np.sqrt(pts1 @ pts1))

sim = pts1 @ pts2 / (np.sqrt(pts1 @ pts1) * np.sqrt(pts2 @ pts2))
sim_sk = cosine_similarity(pts1.reshape(1, -1), pts2.reshape(1, -1))

print(sim == sim_sk)

# euclidean norm

np.linalg.norm(pts1 - pts2)

np.sqrt((pts1 ** 2 + pts2 ** 2).sum())
